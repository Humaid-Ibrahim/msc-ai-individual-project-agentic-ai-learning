# Credential security

## Running code that uses OpenAI

The hint-generation scripts accept OpenAI credentials only from
`OPENAI_API_KEY` or from a programmatic secret-manager injection. They reject a
missing or blank key before importing or constructing the OpenAI client. Do not
pass credentials on a command line, place them in source code, or commit a local
`.env` file.

For an interactive shell, read and export the key without putting its value in
shell history, or use your platform's secret manager. `.env.example` lists the
supported variable names but must remain blank:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
```

`OPENAI_BASE_URL` and `OPENAI_ORG` are optional. The scripts do not
automatically load `.env` files.

## Preventing new leaks

Install the staged-change hook once per clone:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run gitleaks
```

The hook is an early warning, not a complete repository audit. Before a release
or security-sensitive push, scan both the current directory and every reachable
Git ref with Gitleaks:

```bash
gitleaks dir --config .gitleaks.toml --redact --verbose .
gitleaks git --config .gitleaks.toml --redact --verbose --log-opts="--all" .
```

CI performs the full-history command from a complete checkout. The repository
configuration extends every built-in detector and contains one narrow exception
for a known analytics label in two saved notebooks; it does not allow the
compromised credential or its file paths. Keep `--redact` enabled whenever
scanner output might be logged or shared.

## Remediating the previously exposed OpenAI key

Removing a value from the latest files does not remove it from Git history.
Complete these steps as a coordinated security maintenance operation:

1. Revoke the exposed key in the provider account immediately, create a new key
   only if needed, and review provider usage, billing, and audit logs for abuse.
2. Pause repository pushes and coordinate with every collaborator. Commit the
   source-level environment-only fix before taking the cleanup clone.
3. Use Git 2.36 or newer and `git-filter-repo` 2.47 or newer. Work in a fresh,
   disposable clone; do not rewrite a dirty or long-lived working copy.
4. Create a permissions-restricted replacement file outside the repository. Put
   one line in it matching the compromised OpenAI token shape and replacing it
   with a non-secret marker, for example:

   ```text
   regex:sk-proj-[A-Za-z0-9_-]+==>***REMOVED***
   ```

5. In the fresh clone, rewrite all fetched refs:

   ```bash
   git filter-repo \
     --sensitive-data-removal \
     --replace-text /secure/path/replacements.txt
   ```

6. Inspect `.git/filter-repo/changed-refs`, the rewritten commits, and any LFS
   warnings. Run both Gitleaks scans above and `git fsck --full --strict`. Do not
   proceed while any credential finding remains.
7. Restore the `origin` remote if `git-filter-repo` removed it. Temporarily adjust
   branch protection only during the maintenance window, then use
   `git push --force --mirror origin`. Resolve every rejected ref other than
   GitHub's read-only pull-request refs and restore protections immediately.
8. Contact GitHub Support to remove cached commit views and pull-request refs and
   to run server-side garbage collection. Coordinate separately with fork owners.
9. Require collaborators to make fresh clones. Pulling and pushing from an old
   clone can reintroduce the compromised history. Destroy the replacement file
   and retire old clones only after verified migration.

The current project working copy contains unrelated uncommitted work and
tool-managed refs. Preserve that work separately and migrate it into a fresh
post-rewrite clone after scanning it; never use this working copy as the history
rewrite source.

See GitHub's sensitive-data removal guidance and the `git-filter-repo` manual for
the authoritative operational details:

- https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
- https://github.com/newren/git-filter-repo/blob/main/Documentation/git-filter-repo.txt
