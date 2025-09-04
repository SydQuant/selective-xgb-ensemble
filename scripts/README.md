# Scripts Directory

This directory contains automation scripts for the XGBoost Ensemble Trading System.

## auto_wrapup.py

A comprehensive automation script that handles session wrap-up tasks including:
- Analyzing recent code changes
- Updating CLAUDE.md documentation 
- Git operations (staging, committing, pushing)
- Session compacting (planned feature)

### Usage Examples

```bash
# Basic usage - auto-detect changes and commit
python scripts/auto_wrapup.py

# Use with Anaconda Python
~/anaconda3/python.exe scripts/auto_wrapup.py

# Custom commit message
python scripts/auto_wrapup.py --message "Fix XGBoost hyperparameter optimization bug"

# Dry run to see what would happen
python scripts/auto_wrapup.py --dry-run --verbose

# Skip pushing to remote
python scripts/auto_wrapup.py --no-push

# Skip CLAUDE.md updates
python scripts/auto_wrapup.py --no-claude-update

# Verbose output for debugging
python scripts/auto_wrapup.py --verbose
```

### Features

**Intelligent Change Detection**:
- Identifies modified, untracked, and staged files
- Categorizes changes (algorithm, documentation, configuration, etc.)
- Analyzes Python files for new CLI parameters, functions, and classes

**Smart Commit Messages**:
- Auto-generates descriptive commit messages based on detected changes
- Handles different change patterns (new features, bug fixes, documentation updates)
- Includes standard Claude Code footer automatically

**CLAUDE.md Updates**:
- Updates CLI parameters section with new command-line options
- Adds new significant files to critical components section  
- Maintains documentation consistency with codebase changes

**Error Handling**:
- Graceful handling of git command failures
- Detailed logging with timestamp and severity levels
- Dry-run mode for safe testing

### Options

- `--message MSG` - Custom commit message (overrides auto-generation)
- `--dry-run` - Show what would be done without executing
- `--no-push` - Stage and commit but don't push to remote
- `--no-compact` - Skip session compacting step
- `--no-claude-update` - Skip CLAUDE.md documentation updates
- `--verbose` - Enable detailed logging output

### Integration with Development Workflow

The script is designed to be run at the end of development sessions to:

1. **Clean up loose ends** - Ensure all changes are properly committed
2. **Maintain documentation** - Keep CLAUDE.md current with code changes  
3. **Standardize commits** - Consistent commit message format and metadata
4. **Reduce manual overhead** - Automate repetitive end-of-session tasks

### Requirements

- Git repository (must be run from repository root)
- Python 3.6+ with standard library
- Git configured for the repository

### Return Codes

- `0` - Success
- `1` - Error (failed git operations, missing dependencies, etc.)