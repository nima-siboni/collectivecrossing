# Local Deployment Setup

This guide shows you how to deploy your documentation locally without using GitHub Actions.

## 🚀 Quick Setup

### 1. **Build and Deploy in One Command**

```bash
# Deploy directly to GitHub Pages
./scripts/docs.sh deploy
```

This command will:
- Build your documentation
- Deploy it to the `gh-pages` branch
- Make it available at `https://nima-siboni.github.io/collectivecrossing/`

### 2. **Manual Steps (if needed)**

If you prefer to do it step by step:

```bash
# Build the documentation
uv run mkdocs build

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

## 🛠️ Local Development

### Using the Documentation Script

```bash
# Start local development server
./scripts/docs.sh serve

# Build documentation
./scripts/docs.sh build

# Clean build directory
./scripts/docs.sh clean

# Deploy to GitHub Pages
./scripts/docs.sh deploy
```

### Manual Commands

```bash
# Start local server
uv run mkdocs serve

# Build documentation
uv run mkdocs build

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

## 📁 Project Structure

```
collectivecrossing/
├── docs/                          # Documentation source files
│   ├── index.md                   # Home page
│   ├── installation.md            # Installation guide
│   ├── usage.md                   # Usage guide
│   ├── development.md             # Development guide
│   ├── features.md                # Features overview
│   ├── setup_local_deployment.md  # This guide
│   └── assets/                    # Images and other assets
├── mkdocs.yml                     # MkDocs configuration
└── scripts/docs.sh               # Documentation management script
```

## ⚙️ Configuration

### MkDocs Configuration (`mkdocs.yml`)

The configuration uses minimal settings with Material theme defaults:

- **Material theme** with default styling
- **Navigation structure** for organized content
- **Search functionality** for finding content
- **Responsive design** for mobile devices
- **Clean and simple** appearance

## 🔧 How It Works

1. **`mkdocs build`** - Creates HTML files in the `site/` directory
2. **`mkdocs gh-deploy`** - Pushes the built files to the `gh-pages` branch
3. **GitHub Pages** - Serves the files from the `gh-pages` branch

## 🚨 Troubleshooting

### Common Issues

1. **Permission errors**
   - Make sure you have write access to the repository
   - Check that your Git credentials are set up correctly

2. **Build errors**
   - Check that all dependencies are installed: `uv sync --dev`
   - Verify your `mkdocs.yml` configuration

3. **Deployment fails**
   - Ensure you're on the `main` branch
   - Check that you have the latest changes committed

### Getting Help

- Check the [MkDocs documentation](https://www.mkdocs.org/)
- Review the [Material theme documentation](https://squidfunk.github.io/mkdocs-material/)
- Run `./scripts/docs.sh help` for command options

## 📝 Workflow

### Typical Development Workflow

1. **Make changes** to your documentation files
2. **Test locally** with `./scripts/docs.sh serve`
3. **Build and deploy** with `./scripts/docs.sh deploy`
4. **Your docs are live** at the GitHub Pages URL

### Advantages of Local Deployment

- ✅ **Simple and direct** - No complex CI/CD setup
- ✅ **Fast deployment** - Deploy when you want
- ✅ **Full control** - You control when and what gets deployed
- ✅ **No environment issues** - Works on your local machine
- ✅ **Easy debugging** - Test everything locally first

## 🎉 Success!

Once deployed, your documentation will be:

- ✅ Available at `https://nima-siboni.github.io/collectivecrossing/`
- ✅ Searchable and well-organized
- ✅ Mobile-responsive
- ✅ Easy to maintain and update

Happy documenting! 📚✨
