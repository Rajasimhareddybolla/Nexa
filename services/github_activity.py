"""
GitHubUserActivity
------------------
Fetches and saves a complete summary of a user's GitHub activity for a given time window.
Each activity type is stored in its own structured JSON file inside an output folder.

Features:
- Commits, Pull Requests, Issues, Gists, Events, Repositories Worked On
- Supports optional filtering by specific repositories
- PEP8 compliant, modular, and production-ready
"""

import os
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime


class GitHubUserActivity:
    """Collects comprehensive GitHub user activity within a given date range and saves it neatly."""

    def __init__(
        self,
        username: str,
        start_date: str,
        end_date: str,
        token: Optional[str] = None,
        repos: Optional[List[str]] = None
    ):
        """
        Initialize GitHubUserActivity.

        :param username: GitHub username
        :param start_date: Start date in 'YYYY-MM-DD'
        :param end_date: End date in 'YYYY-MM-DD'
        :param token: GitHub API key (Personal Access Token)
        :param repos: Optional list of repository names to filter (e.g., ['repo1', 'repo2'])
        """
        self.username = username
        self.start_date = start_date
        self.end_date = end_date
        self.repos = [r.lower() for r in repos] if repos else None
        self.api_base = "https://api.github.com"
        self.graphql_url = "https://api.github.com/graphql"

        token = token or os.getenv("GITHUB_TOKEN")
        self.headers = {"Accept": "application/vnd.github+json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        # Prepare output directory
        folder_name = f"github_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = os.path.join(os.getcwd(), "output", folder_name)
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------

    def _get(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub API error {response.status_code} for {url}: {response.text}"
            )
        return response

    def _paginate(self, url: str, params: Optional[Dict] = None) -> List[Dict]:
        results = []
        while url:
            resp = self._get(url, params)
            data = resp.json()
            results.extend(data if isinstance(data, list) else [data])
            link = resp.headers.get("Link", "")
            next_url = None
            if 'rel="next"' in link:
                for part in link.split(","):
                    if 'rel="next"' in part:
                        next_url = part.split(";")[0].strip("<> ")
                        break
            url = next_url
            params = None
        return results

    def _save_json(self, data: Dict | List, filename: str):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Saved {filename} ({len(data) if isinstance(data, list) else '1'} items)")

    # ---------------------------------------------------------------
    # Feature Modules
    # ---------------------------------------------------------------

    def _get_user_repos(self) -> List[Dict]:
        url = f"{self.api_base}/users/{self.username}/repos"
        params = {"per_page": 100, "type": "owner", "sort": "updated"}
        return self._paginate(url, params)

    def _get_commits(self) -> List[Dict]:
        since = f"{self.start_date}T00:00:00Z"
        until = f"{self.end_date}T23:59:59Z"
        repos = self._get_user_repos()
        commits_all = []

        for repo in repos:
            owner, repo_name = repo["owner"]["login"], repo["name"].lower()

            # Filter by repos if provided
            if self.repos and repo_name not in self.repos:
                continue

            url = f"{self.api_base}/repos/{owner}/{repo_name}/commits"
            params = {"author": self.username, "since": since, "until": until}
            try:
                commits = self._paginate(url, params)
            except Exception:
                continue

            for c in commits:
                sha = c["sha"]
                detail_url = f"{self.api_base}/repos/{owner}/{repo_name}/commits/{sha}"
                try:
                    detail = self._get(detail_url).json()
                except Exception:
                    continue

                commits_all.append({
                    "repository": f"{owner}/{repo_name}",
                    "sha": sha,
                    "message": detail["commit"]["message"],
                    "author": detail["commit"]["author"],
                    "committer": detail["commit"]["committer"],
                    "stats": detail.get("stats", {}),
                    "files": detail.get("files", []),
                    "parents": [p["sha"] for p in detail.get("parents", [])],
                })

        self._save_json(commits_all, "commits.json")
        return commits_all

    def _get_pull_requests(self) -> List[Dict]:
        query = f"type:pr author:{self.username} created:{self.start_date}..{self.end_date}"
        url = f"{self.api_base}/search/issues"
        params = {"q": query, "per_page": 50}
        response = self._get(url, params).json()

        prs = [
            {
                "title": item["title"],
                "state": item["state"],
                "created_at": item["created_at"],
                "closed_at": item.get("closed_at"),
                "html_url": item["html_url"],
                "repository_url": item["repository_url"],
            }
            for item in response.get("items", [])
        ]

        self._save_json(prs, "pull_requests.json")
        return prs

    def _get_issues(self) -> List[Dict]:
        query = f"type:issue author:{self.username} created:{self.start_date}..{self.end_date}"
        url = f"{self.api_base}/search/issues"
        params = {"q": query, "per_page": 50}
        response = self._get(url, params).json()

        issues = [
            {
                "title": item["title"],
                "state": item["state"],
                "created_at": item["created_at"],
                "closed_at": item.get("closed_at"),
                "html_url": item["html_url"],
                "repository_url": item["repository_url"],
            }
            for item in response.get("items", [])
        ]

        self._save_json(issues, "issues.json")
        return issues

    def _get_gists(self) -> List[Dict]:
        url = f"{self.api_base}/users/{self.username}/gists"
        gists = self._paginate(url)
        filtered = [
            {
                "id": g["id"],
                "description": g["description"],
                "created_at": g["created_at"],
                "updated_at": g["updated_at"],
                "public": g["public"],
                "url": g["html_url"],
            }
            for g in gists
            if self.start_date <= g["created_at"][:10] <= self.end_date
            or self.start_date <= g["updated_at"][:10] <= self.end_date
        ]

        self._save_json(filtered, "gists.json")
        return filtered

    def _get_events(self) -> List[Dict]:
        url = f"{self.api_base}/users/{self.username}/events"
        events = self._paginate(url)
        filtered = [
            {
                "type": e["type"],
                "repo": e["repo"]["name"],
                "created_at": e["created_at"],
                "payload": e["payload"],
            }
            for e in events
            if self.start_date <= e["created_at"][:10] <= self.end_date
        ]

        self._save_json(filtered, "events.json")
        return filtered

    def _get_repositories_worked_on(self) -> List[str]:
        query = {
            "query": f"""
            {{
              user(login: "{self.username}") {{
                contributionsCollection(from: "{self.start_date}T00:00:00Z", to: "{self.end_date}T23:59:59Z") {{
                  commitContributionsByRepository {{
                    repository {{ nameWithOwner }}
                  }}
                }}
              }}
            }}
            """
        }
        response = requests.post(self.graphql_url, json=query, headers=self.headers)
        if response.status_code != 200:
            return []
        data = response.json()
        repos = (
            data.get("data", {})
            .get("user", {})
            .get("contributionsCollection", {})
            .get("commitContributionsByRepository", [])
        )
        repo_names = [r["repository"]["nameWithOwner"] for r in repos]

        self._save_json(repo_names, "repositories_worked_on.json")
        return repo_names

    # ---------------------------------------------------------------
    # Orchestrator
    # ---------------------------------------------------------------

    def get_user_activity(self) -> Dict:
        """Collect all data and save in separate files. Returns a summary."""
        print(f"\n Collecting GitHub activity for @{self.username} ({self.start_date} → {self.end_date})")

        summary = {
            "username": self.username,
            "date_range": {"start": self.start_date, "end": self.end_date},
            "repositories_filter": self.repos if self.repos else "All repositories",
            "repositories_worked_on": self._get_repositories_worked_on(),
            "commits": len(self._get_commits()),
            "pull_requests": len(self._get_pull_requests()),
            "issues": len(self._get_issues()),
            "gists": len(self._get_gists()),
            "events": len(self._get_events()),
            "output_folder": self.output_dir
        }

        self._save_json(summary, "summary.json")
        print("\n All data successfully saved in:", self.output_dir)
        return summary


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    tracker = GitHubUserActivity(
        username="Rajasimhareddybolla",
        start_date="2025-11-06",
        end_date="2025-11-07",
        token=os.getenv("GITHUB_TOKEN"),  # store your token in environment variable
        repos=["bms-AI"]  # <- custom filtering here
    )

    tracker.get_user_activity()

