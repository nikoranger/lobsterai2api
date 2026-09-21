// Package server 暴露 OpenAI 兼容 HTTP 接口，内部驱动 pool 挑号 + upstream 转发。
package server

import (
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"lobsterai2api/internal/auth"
	"lobsterai2api/internal/pool"
	"lobsterai2api/internal/upstream"
)

// Config handler 依赖。
type Config struct {
	Pool          *pool.Pool
	Upstream      *upstream.Client
	APIKey        string        // 空 = 不鉴权
	MaxRotate     int           // 单请求最多换号次数，默认 3
	HardCooldown  time.Duration // 余额不足冷却，默认 12h
	SoftCooldown  time.Duration // 429 冷却，默认 60s
	EmptyCooldown time.Duration // 空响应冷却，默认 60s；0 = 只换号不冷却
	ErrThreshold  int           // 连续其他错误冷却阈值，默认 3
	ErrCooldown   time.Duration // 错误冷却时长，默认 10m
	RefreshSkew   time.Duration // token 提前刷新窗口，默认 10m
}

// Handler 主路由。
type Handler struct {
	cfg Config
	mux *http.ServeMux
}

// NewHandler 构建 handler。
func NewHandler(cfg Config) *Handler {
	if cfg.MaxRotate <= 0 {
		cfg.MaxRotate = 3
	}
	if cfg.HardCooldown <= 0 {
		cfg.HardCooldown = 12 * time.Hour
	}
	if cfg.SoftCooldown <= 0 {
		cfg.SoftCooldown = 60 * time.Second
	}
	// EmptyCooldown 为 0 时只换号不冷却（由 config 层给默认 60s）
	if cfg.ErrThreshold <= 0 {
		cfg.ErrThreshold = 3
	}
	if cfg.ErrCooldown <= 0 {
		cfg.ErrCooldown = 10 * time.Minute
	}
	if cfg.RefreshSkew <= 0 {
		cfg.RefreshSkew = 10 * time.Minute
	}
	h := &Handler{cfg: cfg, mux: http.NewServeMux()}
	h.mux.HandleFunc("POST /v1/chat/completions", h.withAuth(h.chatCompletions))
	h.mux.HandleFunc("GET /v1/models", h.withAuth(h.models))
	h.mux.HandleFunc("GET /status", h.status)
	h.mux.HandleFunc("GET /healthz", h.healthz)
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

func (h *Handler) withAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.cfg.APIKey != "" {
			authz := r.Header.Get("Authorization")
			if !strings.HasPrefix(authz, "Bearer ") || strings.TrimPrefix(authz, "Bearer ") != h.cfg.APIKey {
				writeOpenAIError(w, http.StatusUnauthorized, "invalid_api_key", "missing or invalid API key")
				return
			}
		}
		next(w, r)
	}
}

// refreshQuota 查一次真实积分并写回池子（只更新数值，不做解冻）。
// tag 用于日志区分触发来源；查询失败返回 false（调用方应继续原有处置）。
func (h *Handler) refreshQuota(acct *auth.Auth, tag string) (int64, bool) {
	start := time.Now()
	remain, _, qerr := h.cfg.Upstream.QuotaUsage(acct)
	cost := time.Since(start).Round(time.Millisecond)
	name := acct.Nickname
	if name == "" {
		name = acct.UID
	}
	if qerr != nil {
		log.Printf("quota[%s] uid=%s nick=%s query failed cost=%v err=%v", tag, acct.UID, name, cost, qerr)
		return 0, false
	}
	h.cfg.Pool.SetCredits(acct.UID, remain)
	log.Printf("quota[%s] uid=%s nick=%s remain=%d cost=%v", tag, acct.UID, name, remain, cost)
	return remain, true
}

// penalizeEmpty 空响应处置：查一次真实积分（很可能是余额耗尽导致空回复），
// 再短冷却让下次挑号避开它。返回传入的 cause 供调用方记 lastErr。
// 注意：这里只用 SetCredits 更新数值，不用 ReenableIfCredits —— 空响应的账号不该被解冻。
func (h *Handler) penalizeEmpty(acct *auth.Auth, cause error, where string) error {
	h.refreshQuota(acct, "empty/"+where)
	log.Printf("empty-response uid=%s cause=%v (cooldown %v)", acct.UID, cause, h.cfg.EmptyCooldown)
	h.cfg.Pool.Cooldown(acct.UID, pool.CoolSoft, h.cfg.EmptyCooldown, "empty response")
	return cause
}

func (h *Handler) healthz(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func (h *Handler) status(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"accounts": h.cfg.Pool.List(),
	})
}

// 静态模型表（动态接口失败时的回退）。
// 2026-08-06 从 GET /api/models/available 实测拉取，共 19 个。
var staticModels = []map[string]any{
	{"id": "deepseek-v4-flash", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "deepseek-v4-pro", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "MiniMax-M3", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "MiniMax-M2.7", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "qwen3.7-max", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "qwen3.7-plus", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "qwen3.6-plus", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "qwen3.5-plus-2026-04-20", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "kimi-k2.7-code", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "kimi-k2.7-code-highspeed", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "kimi-k2.6", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "kimi-k2.5", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "doubao-seed-2-1-pro-260628", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "doubao-seed-2-1-turbo-260628", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "doubao-seed-2-0-code-preview-260215", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "glm-5.2", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "glm-5.1", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "glm-5v-turbo", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
	{"id": "glm-5", "object": "model", "created": 1753600000, "owned_by": "lobsterai", "context_length": 131072},
}

// dynamicModelsCache 动态模型缓存。
var dynamicModelsCache struct {
	sync.RWMutex
	ids     []string
	fetched time.Time
}

const dynamicModelsTTL = time.Hour

// models 返回模型列表：优先动态（缓存 1h），失败回退静态表。
func (h *Handler) models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   h.modelList(),
	})
}

// modelList 动态获取模型 ID 列表并包装成 OpenAI 格式。
func (h *Handler) modelList() []map[string]any {
	if ids := h.fetchDynamicModels(); len(ids) > 0 {
		out := make([]map[string]any, 0, len(ids))
		for _, id := range ids {
			out = append(out, map[string]any{
				"id":             id,
				"object":         "model",
				"created":        1753600000,
				"owned_by":       "lobsterai",
				"context_length": 131072,
			})
		}
		return out
	}
	return staticModels
}

// fetchDynamicModels 从池中任一健康账号拉模型列表，缓存 1h。
func (h *Handler) fetchDynamicModels() []string {
	dynamicModelsCache.RLock()
	if len(dynamicModelsCache.ids) > 0 && time.Since(dynamicModelsCache.fetched) < dynamicModelsTTL {
		ids := dynamicModelsCache.ids
		dynamicModelsCache.RUnlock()
		return ids
	}
	dynamicModelsCache.RUnlock()

	// 缓存过期：挑一个健康账号拉
	acct := h.cfg.Pool.Pick()
	if acct == nil {
		return nil
	}
	ids, err := h.cfg.Upstream.FetchModels(acct)
	if err != nil || len(ids) == 0 {
		return nil
	}
	dynamicModelsCache.Lock()
	dynamicModelsCache.ids = ids
	dynamicModelsCache.fetched = time.Now()
	dynamicModelsCache.Unlock()
	return ids
}

func (h *Handler) chatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request", "read body: "+err.Error())
		return
	}
	var peek struct {
		Stream bool   `json:"stream"`
		Model  string `json:"model"`
	}
	_ = json.Unmarshal(body, &peek)

	tried := map[string]bool{}
	var lastErr error
	for i := 0; i < h.cfg.MaxRotate; i++ {
		acct := h.cfg.Pool.PickExcluding(tried)
		if acct == nil {
			break
		}
		tried[acct.UID] = true

		// token 临近过期 → 先 refresh（失败冷却换号）
		if acct.NeedsRefresh(h.cfg.RefreshSkew) {
			if err := h.cfg.Upstream.RefreshToken(acct); err != nil {
				lastErr = err
				var ue *upstream.Error
				if errors.As(err, &ue) && ue.Kind == upstream.ErrSessionDead {
					h.cfg.Pool.Disable(acct.UID, "refresh session dead")
				} else {
					h.cfg.Pool.Cooldown(acct.UID, pool.CoolErr, h.cfg.ErrCooldown, "refresh: "+err.Error())
				}
				continue
			}
			_ = acct.SaveAtomic()
		}

		rc, status, terr := h.cfg.Upstream.ChatStream(acct, body)
		if terr != nil {
			lastErr = terr
			h.cfg.Pool.NoteError(acct.UID, h.cfg.ErrThreshold, h.cfg.ErrCooldown)
			continue
		}
		if status >= 400 {
			kind := upstream.Classify(status, string(h.cfg.Upstream.LastBody))
			switch kind {
			case upstream.ErrHardCredit:
				// 余额不足：先查一次真实积分写回（避免陈旧快照），再长冷却
				if remain, ok := h.refreshQuota(acct, "hard-credit"); ok && remain > 0 {
					log.Printf("quota[hard-credit] uid=%s: classified as no-credit but %d remain — 可能是关键词误判", acct.UID, remain)
				}
				h.cfg.Pool.Cooldown(acct.UID, pool.CoolHard, h.cfg.HardCooldown, "余额不足")
				lastErr = &upstream.Error{Kind: kind, Status: status, Msg: string(h.cfg.Upstream.LastBody)}
				continue
			case upstream.ErrSoftRate:
				h.cfg.Pool.Cooldown(acct.UID, pool.CoolSoft, h.cfg.SoftCooldown, "429 rate limit")
				lastErr = &upstream.Error{Kind: kind, Status: status, Msg: string(h.cfg.Upstream.LastBody)}
				continue
			case upstream.ErrSessionDead:
				h.cfg.Pool.Disable(acct.UID, "session dead")
				lastErr = &upstream.Error{Kind: kind, Status: status, Msg: string(h.cfg.Upstream.LastBody)}
				continue
			case upstream.ErrNotFound:
				// 404 短冷却不累计 errCount（防雪崩）
				h.cfg.Pool.Cooldown(acct.UID, pool.CoolSoft, h.cfg.SoftCooldown, "upstream 404")
				lastErr = &upstream.Error{Kind: kind, Status: status, Msg: string(h.cfg.Upstream.LastBody)}
				continue
			default:
				// 轮转下一个账号，不直接返回（防雪崩）
				h.cfg.Pool.NoteError(acct.UID, h.cfg.ErrThreshold, h.cfg.ErrCooldown)
				lastErr = &upstream.Error{Kind: kind, Status: status, Msg: string(h.cfg.Upstream.LastBody)}
				continue
			}
		}
		// 流式：预读未拿到任何实质内容时，还没写过字节 → 换号重试
		if peek.Stream {
			sawContent, serr := upstream.Stream(w, rc)
			rc.Close()
			if serr != nil {
				log.Printf("chat_stream uid=%s: %v", acct.UID, serr)
				if sawContent {
					return // 已向客户端输出，无法再换号
				}
				lastErr = serr
				h.cfg.Pool.NoteError(acct.UID, h.cfg.ErrThreshold, h.cfg.ErrCooldown)
				continue
			}
			if !sawContent {
				lastErr = h.penalizeEmpty(acct, errors.New("empty upstream stream"), "stream")
				continue
			}
			h.cfg.Pool.NoteSuccess(acct.UID)
			return
		}
		// 非流式：空响应/业务错误 → 刷新积分 + 冷却 + 换号重试
		resp, err := upstream.Aggregate(rc, peek.Model)
		rc.Close()
		if err != nil {
			lastErr = h.penalizeEmpty(acct, err, "aggregate")
			continue
		}
		h.cfg.Pool.NoteSuccess(acct.UID)
		writeJSON(w, http.StatusOK, resp)
		return
	}
	msg := "all accounts unavailable (cooling/disabled)"
	if lastErr != nil {
		msg += ": " + lastErr.Error()
	}
	writeOpenAIError(w, http.StatusServiceUnavailable, "no_healthy_account", msg)
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v any) {
	raw, _ := json.Marshal(v)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(raw)
}

func writeOpenAIError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "api_error",
			"code":    code,
		},
	})
}
