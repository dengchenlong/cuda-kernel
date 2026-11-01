#!/bin/sh
# 推送 main 并校验远端 SHA —— 直连被墙时自动走本机代理。
# 用法：scripts/push.sh [branch]        分支默认 main
# 成功最后一行打印 SYNC_OK <sha>，失败打印 SYNC_FAIL <原因> 并返回非零。
# 为什么存在：git push 会因网络重置静默失败，2026-10-27 有 commit 在本地躺了 5 小时没人知道。
# 2026-11-02 补：无人值守（cron）下 Git Credential Manager 想开 /dev/tty 提示凭据 →
#   "/dev/tty: No such device or address" 后 push 根本没发出去，而旧版 push_once 的退出码
#   来自 `| tail -3`（恒 0），于是打出 push_ok=1 的假成功。现在：
#   ① GIT_TERMINAL_PROMPT=0 禁止凭据提示；② gh 已登录时用 `gh auth git-credential`
#   （-c credential.helper= 先清空 GCM，避免它抢答）；③ push 退出码取自 git 本身。
set -u
export GIT_TERMINAL_PROMPT=0
BRANCH="${1:-main}"
cd "$(git rev-parse --show-toplevel)" || exit 2

PROXY="$(git config --get http.proxy || true)"
if [ -z "$PROXY" ]; then
  for p in 7897 7890 10809 1080; do
    if curl -s -o /dev/null --max-time 5 -x "http://127.0.0.1:$p" https://github.com; then
      PROXY="http://127.0.0.1:$p"; break
    fi
  done
fi

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  # 用 GIT_CONFIG_* 环境变量注入（不用 -c 参数）：这样 timeout 仍能直接 exec git
  GIT_CONFIG_COUNT=2;                          export GIT_CONFIG_COUNT
  GIT_CONFIG_KEY_0=credential.helper;          export GIT_CONFIG_KEY_0
  GIT_CONFIG_VALUE_0=;                         export GIT_CONFIG_VALUE_0
  GIT_CONFIG_KEY_1=credential.helper;          export GIT_CONFIG_KEY_1
  GIT_CONFIG_VALUE_1='!gh auth git-credential'; export GIT_CONFIG_VALUE_1
fi

LOG="$(git rev-parse --show-toplevel)/build/_push.log"
mkdir -p "$(dirname "$LOG")"

push_once() {
  if [ -n "$PROXY" ]; then
    timeout 180 git -c http.proxy="$PROXY" push origin "$BRANCH" >"$LOG" 2>&1
  else
    timeout 180 git push origin "$BRANCH" >"$LOG" 2>&1
  fi
  rc=$?
  tail -3 "$LOG"
  return $rc
}

ok=0
for i in 1 2 3; do
  echo "[push] 第 $i 次尝试 (proxy=${PROXY:-直连})"
  if push_once; then ok=1; break; fi
  sleep 5
done

LOCAL="$(git rev-parse HEAD)"
if [ -n "$PROXY" ]; then
  REMOTE="$(timeout 90 git -c http.proxy="$PROXY" ls-remote origin "refs/heads/$BRANCH" | cut -f1)"
else
  REMOTE="$(timeout 90 git ls-remote origin "refs/heads/$BRANCH" | cut -f1)"
fi

if [ "$LOCAL" = "$REMOTE" ]; then
  echo "SYNC_OK $(echo "$LOCAL" | cut -c1-7)"
  exit 0
fi
echo "SYNC_FAIL local=$(echo "$LOCAL" | cut -c1-7) remote=$(echo "${REMOTE:-<取不到>}" | cut -c1-7) push_ok=$ok"
exit 1
