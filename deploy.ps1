# DEPLOY.ps1 - One-Click GitHub Deployment
param(
    [string]$RepoName,
    [string]$Description = "An intelligent RAG system powered by Google's Gemini 2.0 Flash Thinking, Qdrant vector storage, and Agno agent orchestration."
)

# 1. GitHub Connection
$GitHubUser = "Abdulraqib20"
git remote set-url origin "https://github.com/$GitHubUser/$RepoName.git"

# 2. Generate Professional README
@"
# 🚀 $RepoName

$Description

## ✨ Features
- 📄 Multi-PDF & Web Content Processing
- 🧠 Gemini 2.0 Flash Integration
- 🔍 Hybrid Vector Search
- 🌐 Web Search Fallback

## 🛠️ Installation
```bash
git clone https://github.com/$GitHubUser/$RepoName.git
cd $RepoName
pip install -r requirements.txt
```

## ⚙️ Configuration
Create .env file with: <br>
QDRANT_URL="your-url" <br>
QDRANT_API_KEY="your-key" <br>
GOOGLE_API_KEY="google-key" <br>
EXAAI_API_KEY="exa-key" <br>

## 📧 Support
Report issues at: https://github.com/$GitHubUser/$RepoName/issues
"@ | Out-File README.md -Encoding UTF8

# 3. Commit Changes
git add --all
git commit -m "🚀 Deploy: $RepoName"

# 4. Force Push
git push -u origin main --force

Write-Host "✅ Deployment Successful: https://github.com/$GitHubUser/$RepoName" -ForegroundColor Green


#------------------------------------
# Steps to Deploy
#------------------------------------

# 1. Remove git repository
# rm -Recurse -Force .git OR Remove-Item -Recurse -Force .git

# 2. initialize with Main Branch
# git init -b main

# 3. Show current branch
# git branch --show-current  # Should output "main"

# 4. Set remote URL 
# git remote add origin https://github.com/Abdulraqib20/Agentic-RAG-with-Gemini-2.0-Flash

# 5. Run the below in PowerShell:
# ./deploy.ps1 -RepoName "Agentic-RAG-with-Gemini-2.0-Flash" -Description "An intelligent RAG system powered by Google's Gemini 2.0 Flash Thinking, Qdrant vector storage, and Agno agent orchestration. Upload documents, process web pages, and get AI-assisted answers with advanced query rewriting and web search capabilities."

# Credential Caching (One-Time Setup):
# git config --global credential.helper wincred


