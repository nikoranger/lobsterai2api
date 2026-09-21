// sse.go 处理上游 SSE 流：聚合成单个 OpenAI 响应，或透传给客户端。
package upstream

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Aggregate 读取完整 SSE 流，聚合 delta.content 为单个 OpenAI chat.completion 响应。
// 分片/半行由 bufio.Reader.ReadString 处理；遇到 "data: [DONE]" 结束。
// tool_calls 以流式 delta 到达（按 index 合并：首片带 id/type/name，后续只带 arguments 片段）。
//
// fallbackModel：上游未返回 model 时的兜底（取客户端请求体里的 model）。
// 空流 / 无任何实质内容 / 业务信封 code!=0 均返回 error，供调用方换号重试。
func Aggregate(r io.Reader, fallbackModel string) (map[string]any, error) {
	br := bufio.NewReaderSize(r, 64*1024)
	var (
		id, model     string
		created       float64
		content       strings.Builder
		reasoning     strings.Builder
		role          = "assistant"
		finishReason  = "stop"
		usage         map[string]any
		gotAnyContent bool
		toolCalls     = map[int]map[string]any{}
		toolOrder     []int
		parsed        int // 成功解析的 data 分片数
		rawHead       strings.Builder
	)
	head := func(line string) {
		if rawHead.Len() < 200 {
			rawHead.WriteString(line)
			rawHead.WriteString(" | ")
		}
	}
	for {
		line, err := br.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		head(line)
		// 兼容 "data: {...}" 与 "data:{...}"（龙虾上游实测无空格）
		if strings.HasPrefix(line, "data:") {
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "[DONE]" {
				// drain nothing; done
			} else {
				var chunk map[string]any
				if json.Unmarshal([]byte(payload), &chunk) == nil {
					parsed++
					// 嵌套信封兼容：{code,msg,data:{...}} → 取内层
					if inner, ok := chunk["data"].(map[string]any); ok {
						if _, has := chunk["choices"]; !has {
							chunk = inner
						}
					}
					// 上游把业务错误塞进 200 + SSE 信封的情况
					if code, ok := chunk["code"].(float64); ok && code != 0 {
						msg, _ := chunk["msg"].(string)
						return nil, fmt.Errorf("upstream business error code=%d msg=%s", int(code), truncate(msg, 120))
					}
					if v, ok := chunk["id"].(string); ok && id == "" {
						id = v
					}
					if v, ok := chunk["model"].(string); ok && model == "" {
						model = v
					}
					if v, ok := chunk["created"].(float64); ok && created == 0 {
						created = v
					}
					if u, ok := chunk["usage"].(map[string]any); ok {
						usage = u
					}
					if ch, ok := chunk["choices"].([]any); ok {
						for _, ci := range ch {
							c, _ := ci.(map[string]any)
							if c == nil {
								continue
							}
							if fr, ok := c["finish_reason"].(string); ok && fr != "" {
								finishReason = fr
							}
							if delta, ok := c["delta"].(map[string]any); ok {
								if r2, ok := delta["role"].(string); ok && r2 != "" {
									role = r2
								}
								if txt, ok := delta["content"].(string); ok {
									content.WriteString(txt)
									gotAnyContent = true
								}
								if rc, ok := delta["reasoning_content"].(string); ok {
									reasoning.WriteString(rc)
								}
								if tcs, ok := delta["tool_calls"].([]any); ok {
									for _, tc := range tcs {
										call, ok := tc.(map[string]any)
										if !ok {
											continue
										}
										idx := 0
										if v, ok := call["index"].(float64); ok {
											idx = int(v)
										}
										merged, seen := toolCalls[idx]
										if !seen {
											merged = map[string]any{"index": idx}
											toolCalls[idx] = merged
											toolOrder = append(toolOrder, idx)
										}
										mergeToolCallDelta(merged, call)
									}
								}
							}
							// 有的上游把完整消息放在 message 里（非 delta）
							if msg, ok := c["message"].(map[string]any); ok && !gotAnyContent {
								if txt, ok := msg["content"].(string); ok {
									content.WriteString(txt)
								}
							}
						}
					}
				}
			}
		}
		if err == io.EOF {
			break
		}
	}
	// 空流判定：没有任何可解析分片 → 换号重试（上游可能 200 + 非 SSE body）
	if parsed == 0 {
		return nil, fmt.Errorf("empty upstream stream: no parsable chunk, head=%s", truncate(rawHead.String(), 200))
	}
	// 有分片但无任何实质内容（content/reasoning/tool_calls 全空）→ 同样换号重试
	if content.Len() == 0 && reasoning.Len() == 0 && len(toolOrder) == 0 {
		return nil, fmt.Errorf("empty upstream content: %d chunk(s) with no content, head=%s", parsed, truncate(rawHead.String(), 200))
	}
	if id == "" {
		id = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}
	if created == 0 {
		created = float64(time.Now().Unix())
	}
	if model == "" {
		model = fallbackModel
	}
	message := map[string]any{
		"role":    role,
		"content": content.String(),
	}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}
	if len(toolOrder) > 0 {
		sortInts(toolOrder)
		calls := make([]map[string]any, 0, len(toolOrder))
		for _, idx := range toolOrder {
			calls = append(calls, toolCalls[idx])
		}
		message["tool_calls"] = calls
	}
	resp := map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": int64(created),
		"model":   model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
	}
	if usage != nil {
		resp["usage"] = usage
	}
	return resp, nil
}

// mergeToolCallDelta 把流式 tool_call 片段合并到累计对象：
// id/type/function.name 直覆盖（后续分片通常缺省），function.arguments 拼接。
func mergeToolCallDelta(merged, delta map[string]any) {
	if v, ok := delta["id"].(string); ok && v != "" {
		merged["id"] = v
	}
	if v, ok := delta["type"].(string); ok && v != "" {
		merged["type"] = v
	}
	df, _ := delta["function"].(map[string]any)
	if df == nil {
		return
	}
	mf, _ := merged["function"].(map[string]any)
	if mf == nil {
		mf = map[string]any{}
		merged["function"] = mf
	}
	if v, ok := df["name"].(string); ok && v != "" {
		mf["name"] = v
	}
	if v, ok := df["arguments"].(string); ok && v != "" {
		if prev, _ := mf["arguments"].(string); prev != "" {
			mf["arguments"] = prev + v
		} else {
			mf["arguments"] = v
		}
	}
}

// sortInts 升序排序（避免引 sort 包只为三行）。
func sortInts(a []int) {
	for i := 0; i < len(a)-1; i++ {
		for j := i + 1; j < len(a); j++ {
			if a[j] < a[i] {
				a[i], a[j] = a[j], a[i]
			}
		}
	}
}

// ssePayload 提取一行 SSE 的 data 负载；非 data 行返回空串。
func ssePayload(line string) string {
	trimmed := strings.TrimSpace(strings.TrimRight(line, "\r\n"))
	if !strings.HasPrefix(trimmed, "data:") {
		return ""
	}
	return strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
}

// chunkHasContent 判断一行 SSE 是否携带实质内容（content / reasoning_content / tool_calls）。
// 只带 role、usage、finish_reason 的分片不算内容 —— 这正是"连续空响应"的特征。
func chunkHasContent(line string) bool {
	payload := ssePayload(line)
	if payload == "" || payload == "[DONE]" {
		return false
	}
	var chunk map[string]any
	if json.Unmarshal([]byte(payload), &chunk) != nil {
		return false
	}
	if inner, ok := chunk["data"].(map[string]any); ok {
		if _, has := chunk["choices"]; !has {
			chunk = inner
		}
	}
	chs, _ := chunk["choices"].([]any)
	for _, ci := range chs {
		c, _ := ci.(map[string]any)
		if c == nil {
			continue
		}
		if d, ok := c["delta"].(map[string]any); ok {
			if s, _ := d["content"].(string); s != "" {
				return true
			}
			if s, _ := d["reasoning_content"].(string); s != "" {
				return true
			}
			if tcs, ok := d["tool_calls"].([]any); ok && len(tcs) > 0 {
				return true
			}
		}
		if m, ok := c["message"].(map[string]any); ok {
			if s, _ := m["content"].(string); s != "" {
				return true
			}
			if tcs, ok := m["tool_calls"].([]any); ok && len(tcs) > 0 {
				return true
			}
		}
	}
	return false
}

// Stream 透传上游 SSE 到 w（每行 flush），保证至少写一个 [DONE]。
//
// 预读策略：先把分片攒在内存里，直到出现第一个带实质内容的分片才写 header 并冲刷。
// 这样"上游 200 但连续空响应"时本函数一个字节都没写过，返回 false，
// 调用方可以安全换号重试（而不是把空流推给客户端）。
// 返回是否收到过实质内容。
func Stream(w http.ResponseWriter, r io.Reader) (bool, error) {
	br := bufio.NewReaderSize(r, 64*1024)

	// ---- 预读阶段 ----
	var pending []string
	var pendingBytes int
	sawContent := false
	sawDone := false
	for !sawContent && pendingBytes < 64<<10 {
		line, err := br.ReadString('\n')
		if line != "" {
			pending = append(pending, line)
			pendingBytes += len(line)
			if ssePayload(line) == "[DONE]" {
				sawDone = true
			}
			if chunkHasContent(line) {
				sawContent = true
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return false, err // 尚未写过任何字节，调用方可以换号
		}
	}
	// 流已结束（或只有心跳/空分片）仍无内容 → 判定空响应
	if !sawContent && pendingBytes < 64<<10 {
		return false, nil
	}

	// ---- 正常透传 ----
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	fl, _ := w.(http.Flusher)
	write := func(s string) error {
		if _, werr := io.WriteString(w, s); werr != nil {
			return werr
		}
		if fl != nil {
			fl.Flush()
		}
		return nil
	}
	for _, line := range pending {
		if err := write(line); err != nil {
			return true, err
		}
	}
	for {
		line, err := br.ReadString('\n')
		if line != "" {
			if ssePayload(line) == "[DONE]" {
				sawDone = true
			}
			if err := write(line); err != nil {
				return true, err
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return true, err
		}
	}
	if !sawDone {
		if err := write("data: [DONE]\n\n"); err != nil {
			return true, err
		}
	}
	return true, nil
}
