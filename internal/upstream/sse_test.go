package upstream

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAggregateNormal(t *testing.T) {
	in := strings.Join([]string{
		`data: {"id":"c1","model":"deepseek-v4-pro","created":1,"choices":[{"delta":{"role":"assistant","content":"你"}}]}`,
		``,
		`data: {"choices":[{"delta":{"content":"好"}}]}`,
		``,
		`data: {"choices":[{"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	resp, err := Aggregate(strings.NewReader(in), "fallback-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp["model"] != "deepseek-v4-pro" {
		t.Fatalf("model = %v", resp["model"])
	}
	ch := resp["choices"].([]any)[0].(map[string]any)
	if got := ch["message"].(map[string]any)["content"]; got != "你好" {
		t.Fatalf("content = %v", got)
	}
}

func TestAggregateFallbackModel(t *testing.T) {
	in := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
	resp, err := Aggregate(strings.NewReader(in), "req-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp["model"] != "req-model" {
		t.Fatalf("model = %v, want req-model", resp["model"])
	}
}

func TestAggregateEmptyStream(t *testing.T) {
	// 上游 200 + 非 SSE 业务信封：必须报错以便换号
	in := `{"code":40100,"msg":"session expired"}`
	if _, err := Aggregate(strings.NewReader(in), ""); err == nil {
		t.Fatal("want error for empty/non-SSE stream")
	}
}

func TestAggregateBusinessErrorInSSE(t *testing.T) {
	in := "data: {\"code\":500,\"msg\":\"upstream exploded\"}\n\n"
	if _, err := Aggregate(strings.NewReader(in), ""); err == nil {
		t.Fatal("want error for business code != 0")
	}
}

func TestAggregateNoContent(t *testing.T) {
	// 有分片但只有 role / finish_reason，没有任何实质内容
	in := strings.Join([]string{
		`data: {"choices":[{"delta":{"role":"assistant","content":""}}]}`,
		``,
		`data: {"choices":[{"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	if _, err := Aggregate(strings.NewReader(in), ""); err == nil {
		t.Fatal("want error for content-less stream")
	}
}

func TestAggregateNestedEnvelope(t *testing.T) {
	in := "data: {\"code\":0,\"data\":{\"model\":\"m1\",\"choices\":[{\"delta\":{\"content\":\"x\"}}]}}\n\n"
	resp, err := Aggregate(strings.NewReader(in), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp["model"] != "m1" {
		t.Fatalf("model = %v", resp["model"])
	}
}

func TestStreamEmptyWritesNothing(t *testing.T) {
	in := strings.Join([]string{
		`data: {"choices":[{"delta":{"role":"assistant","content":""}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	rec := httptest.NewRecorder()
	saw, err := Stream(rec, strings.NewReader(in))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if saw {
		t.Fatal("sawContent = true, want false")
	}
	if rec.Body.Len() != 0 {
		t.Fatalf("wrote %d bytes, want 0 (must stay retryable)", rec.Body.Len())
	}
	if rec.Header().Get("Content-Type") != "" {
		t.Fatalf("header already set: %v", rec.Header())
	}
}

func TestStreamWithContent(t *testing.T) {
	in := strings.Join([]string{
		`data: {"choices":[{"delta":{"role":"assistant"}}]}`,
		``,
		`data: {"choices":[{"delta":{"content":"hi"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	rec := httptest.NewRecorder()
	saw, err := Stream(rec, strings.NewReader(in))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !saw {
		t.Fatal("sawContent = false, want true")
	}
	// 预读缓冲必须原样吐出，不能丢首片
	if !strings.Contains(rec.Body.String(), `"role":"assistant"`) {
		t.Fatalf("buffered head lost: %q", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"content":"hi"`) {
		t.Fatalf("content lost: %q", rec.Body.String())
	}
}
