"""공개 레포 → 운영 레포 분석 입력 텍스트 (SPEC §4.1: README + 트리 200줄 + 샘플 소스 8개).

운영 `GitHubRepoSourceExtractor` 를 그대로 쓴다. GitHub API 대신 고정 커밋의 로컬 클론을
httpx.MockTransport 로 같은 응답 모양으로 돌려준다. 그래서 파일 선택·조립 로직은 운영과 같다.
지원자 토큰이 없으므로 운영의 '지원자 기여' 절은 생기지 않는다 (공개 레포 fallback 경로와 같음).

개인정보: AUTHORS·CONTRIBUTORS·CODEOWNERS·MAINTAINERS·.mailmap 은 트리에서 빼고,
README·소스 본문의 이메일 주소는 `[email]` 로 가린다. 바꾼 내용은 메타데이터에 남긴다.
"""

from __future__ import annotations

import asyncio
import base64
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

REPO_DIR = Path.home() / "rag-lab" / "repos"
_PII_FILES = re.compile(r"(^|/)(AUTHORS|CONTRIBUTORS|CODEOWNERS|MAINTAINERS|\.mailmap)(\.[A-Za-z]+)?$", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _git(repo: Path, *args: str, binary: bool = False) -> Any:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode("utf-8", errors="replace")


def ensure_clone(full_name: str, commit: str) -> Path:
    dest = REPO_DIR / full_name.replace("/", "__")
    if not dest.exists():
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                        f"https://github.com/{full_name}.git", str(dest)], check=True)
    try:
        _git(dest, "cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        _git(dest, "fetch", "-q", "origin", commit)
    return dest


class _LocalGitHub:
    def __init__(self, full_name: str, repo: Path, commit: str, branch: str, description: str) -> None:
        self.full_name, self.repo, self.commit, self.branch, self.description = full_name, repo, commit, branch, description
        self.paths = [p for p in _git(repo, "ls-tree", "-r", "--name-only", "--full-tree", commit).splitlines() if p]
        self.excluded = [p for p in self.paths if _PII_FILES.search(p)]
        self.paths = [p for p in self.paths if not _PII_FILES.search(p)]
        self.redactions = 0

    def _blob(self, path: str) -> bytes:
        data = _git(self.repo, "show", f"{self.commit}:{path}", binary=True)
        text = data.decode("utf-8", errors="replace")
        red, n = _EMAIL.subn("[email]", text)
        self.redactions += n
        return red.encode("utf-8")

    def handle(self, request: httpx.Request) -> httpx.Response:
        prefix = f"/repos/{self.full_name}"
        path = request.url.path
        if path == prefix:
            return httpx.Response(200, json={"default_branch": self.branch, "description": self.description})
        if path == f"{prefix}/readme":
            roots = [p for p in self.paths if "/" not in p and p.lower().startswith("readme")]
            if not roots:
                return httpx.Response(404, json={})
            pick = sorted(roots, key=lambda p: (p.lower() != "readme.md", p))[0]
            return httpx.Response(200, json={"encoding": "base64", "content": base64.b64encode(self._blob(pick)).decode()})
        if path.startswith(f"{prefix}/git/trees/"):
            return httpx.Response(200, json={"tree": [{"path": p, "type": "blob"} for p in self.paths], "truncated": False})
        if path.startswith(f"{prefix}/contents/"):
            rel = path[len(f"{prefix}/contents/"):]
            if rel not in self.paths:
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"encoding": "base64", "content": base64.b64encode(self._blob(rel)).decode()})
        return httpx.Response(404, json={})


def build_repo_text(full_name: str, commit: str, branch: str, description: str) -> tuple[str, dict[str, Any]]:
    from ai_server.analyzer.sources.github_repo import GitHubRepoSourceExtractor
    from ai_server.config.settings import Settings

    s = Settings(_env_file=None)
    repo = ensure_clone(full_name, commit)
    gh = _LocalGitHub(full_name, repo, commit, branch, description)
    client = httpx.AsyncClient(base_url="https://api.github.local", transport=httpx.MockTransport(gh.handle))
    ex = GitHubRepoSourceExtractor(api_base_url="https://api.github.local", max_files=s.repo_max_source_files,
                                   max_file_bytes=s.repo_max_source_file_bytes, client=client)

    async def run() -> Any:
        try:
            return await ex.extract(full_name)
        finally:
            await client.aclose()

    src = asyncio.run(run())
    meta = dict(src.metadata)
    meta.update(pii_excluded_files=gh.excluded, email_redactions=gh.redactions, commit=commit,
                tree_size_after_exclusion=len(gh.paths))
    return src.text, meta
