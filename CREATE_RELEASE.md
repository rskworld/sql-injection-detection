# How to Create a GitHub Release

The code and tag (v1.0.0) have been successfully pushed to GitHub. To create a release:

## Option 1: Using GitHub Web Interface (Recommended)

1. Go to your repository: https://github.com/rskworld/sql-injection-detection
2. Click on **"Releases"** (on the right sidebar or under the repository name)
3. Click **"Create a new release"**
4. Select tag: **v1.0.0**
5. Release title: **v1.0.0 - SQL Injection Detection using NLP**
6. Description: Copy the content from `RELEASE_NOTES_v1.0.0.md`
7. Click **"Publish release"**

## Option 2: Using GitHub CLI

If you have GitHub CLI installed:

```bash
gh release create v1.0.0 --title "v1.0.0 - SQL Injection Detection using NLP" --notes-file RELEASE_NOTES_v1.0.0.md
```

## Option 3: Using GitHub API

You can use curl to create a release via API (requires a personal access token):

```bash
curl -X POST \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/rskworld/sql-injection-detection/releases \
  -d '{
    "tag_name": "v1.0.0",
    "name": "v1.0.0 - SQL Injection Detection using NLP",
    "body": "Release notes here...",
    "draft": false,
    "prerelease": false
  }'
```

## What's Already Done

✅ Code pushed to GitHub  
✅ Tag v1.0.0 created and pushed  
✅ Release notes file created (`RELEASE_NOTES_v1.0.0.md`)

## Next Steps

Just create the release using one of the methods above!

