#!/usr/bin/env python3
"""
Auto Wrap-up Script for XGBoost Ensemble Trading System

This script automates the session wrap-up process by:
1. Analyzing recent changes in the codebase
2. Updating CLAUDE.md documentation to reflect changes
3. Compacting the session (if /compact command exists)
4. Performing git operations (staging, committing, pushing)

Usage:
    python scripts/auto_wrapup.py [options]
    ~/anaconda3/python.exe scripts/auto_wrapup.py [options]

Options:
    --message MSG       Custom commit message
    --dry-run          Show what would be done without executing
    --no-push          Don't push to remote repository
    --no-compact       Skip session compacting
    --no-claude-update Skip CLAUDE.md updates
    --verbose          Verbose output
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class AutoWrapupManager:
    """Manages the automated wrap-up process for the XGBoost trading system."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.root_dir = Path.cwd()
        self.claude_md_path = self.root_dir / "CLAUDE.md"
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log messages with optional verbose output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {level}: "
        
        if level == "VERBOSE" and not self.verbose:
            return
            
        print(f"{prefix}{message}")
    
    def run_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[bool, str]:
        """Execute a command and return success status and output."""
        cmd_str = " ".join(cmd)
        self.log(f"Running: {cmd_str}", "VERBOSE")
        
        # For read-only commands like git status, always execute even in dry-run
        read_only_commands = ["git status", "git log", "git diff"]
        is_read_only = any(cmd_str.startswith(readonly) for readonly in read_only_commands)
        
        if self.dry_run and not is_read_only:
            self.log(f"DRY RUN: Would execute: {cmd_str}")
            return True, ""
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                cwd=self.root_dir
            )
            
            if result.returncode != 0:
                self.log(f"Command failed: {cmd_str}", "ERROR")
                self.log(f"Error: {result.stderr}", "ERROR")
                return False, result.stderr
            
            return True, result.stdout.strip()
        
        except Exception as e:
            self.log(f"Exception running command: {e}", "ERROR")
            return False, str(e)
    
    def get_git_status(self) -> Dict[str, List[str]]:
        """Get current git status - modified, untracked, staged files."""
        status = {"modified": [], "untracked": [], "staged": []}
        
        success, output = self.run_command(["git", "status", "--porcelain"])
        if not success:
            return status
        
        for line in output.split('\n'):
            if not line.strip():
                continue
                
            status_code = line[:2]
            filename = line[3:].strip()
            
            if status_code[0] in ['M', 'A', 'D', 'R', 'C']:
                status["staged"].append(filename)
            elif status_code[1] == 'M':
                status["modified"].append(filename)
            elif status_code == '??':
                status["untracked"].append(filename)
        
        return status
    
    def get_recent_commits(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent commit messages for context."""
        success, output = self.run_command([
            "git", "log", f"--max-count={limit}", 
            "--pretty=format:%H|%s|%ad", "--date=short"
        ])
        
        if not success:
            return []
        
        commits = []
        for line in output.split('\n'):
            if '|' in line:
                hash_val, subject, date = line.split('|', 2)
                commits.append({
                    "hash": hash_val[:8],
                    "subject": subject,
                    "date": date
                })
        
        return commits
    
    def detect_change_types(self, files: Dict[str, List[str]]) -> Set[str]:
        """Analyze files to detect types of changes made."""
        change_types = set()
        all_files = files["modified"] + files["untracked"] + files["staged"]
        
        for file_path in all_files:
            file_lower = file_path.lower()
            
            # Documentation changes
            if file_path.endswith('.md') or 'readme' in file_lower:
                change_types.add("documentation")
            
            # Configuration changes  
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                change_types.add("configuration")
            
            # Script/utility changes
            elif file_path.startswith('scripts/') or file_path.endswith('.py'):
                if 'test' in file_lower or 'analyze' in file_lower:
                    change_types.add("analysis")
                else:
                    change_types.add("code")
            
            # Model/algorithm changes
            elif any(dir_name in file_path for dir_name in ['model/', 'ensemble/', 'opt/']):
                change_types.add("algorithm")
            
            # Data pipeline changes
            elif 'data/' in file_path:
                change_types.add("data_pipeline")
            
            # Evaluation/metrics changes
            elif any(dir_name in file_path for dir_name in ['eval/', 'metrics/', 'cv/']):
                change_types.add("evaluation")
            
            # Artifact/output changes
            elif file_path.startswith('artifacts/') or file_path.startswith('logs/'):
                change_types.add("artifacts")
        
        # Default fallback
        if not change_types and all_files:
            change_types.add("general")
        
        return change_types
    
    def analyze_code_changes(self, files: List[str]) -> Dict[str, any]:
        """Analyze code files for new features, CLI parameters, etc."""
        analysis = {
            "new_cli_params": [],
            "new_functions": [],
            "new_classes": [],
            "modified_configs": [],
            "potential_features": []
        }
        
        cli_param_pattern = r'ap\.add_argument\("--([^"]+)".*?help="([^"]*)"'
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]'
        
        for file_path in files:
            if not file_path.endswith('.py'):
                continue
                
            full_path = self.root_dir / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for new CLI parameters in main.py
                if file_path == 'main.py':
                    cli_matches = re.findall(cli_param_pattern, content)
                    for param, help_text in cli_matches:
                        analysis["new_cli_params"].append({
                            "param": param,
                            "help": help_text,
                            "file": file_path
                        })
                
                # Find new functions
                func_matches = re.findall(function_pattern, content)
                for func_name in func_matches:
                    if not func_name.startswith('_'):  # Skip private functions
                        analysis["new_functions"].append({
                            "name": func_name,
                            "file": file_path
                        })
                
                # Find new classes
                class_matches = re.findall(class_pattern, content)
                for class_name in class_matches:
                    analysis["new_classes"].append({
                        "name": class_name,
                        "file": file_path
                    })
                
            except Exception as e:
                self.log(f"Error analyzing {file_path}: {e}", "ERROR")
        
        return analysis
    
    def update_claude_md(self, status: Dict[str, List[str]], change_types: Set[str], 
                        analysis: Dict[str, any]) -> bool:
        """Update CLAUDE.md with detected changes."""
        if not self.claude_md_path.exists():
            self.log("CLAUDE.md not found, skipping update", "WARNING")
            return False
        
        try:
            with open(self.claude_md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            today = datetime.now().strftime("%B %Y")
            
            # Update completion section if we have significant changes
            if "algorithm" in change_types or "code" in change_types:
                self._update_completed_testing_section(content, change_types, analysis)
            
            # Update CLI parameters if new ones detected
            if analysis.get("new_cli_params"):
                content = self._update_cli_parameters_section(content, analysis["new_cli_params"])
            
            # Update known issues if bug fixes detected
            if any("fix" in f.lower() for f in status["modified"] + status["untracked"]):
                content = self._update_known_issues_section(content, status, today)
            
            # Update critical components if new files added
            new_py_files = [f for f in status["untracked"] if f.endswith('.py')]
            if new_py_files:
                content = self._update_critical_components(content, new_py_files, analysis)
            
            # Only write if content changed
            if content != original_content:
                if self.dry_run:
                    self.log("DRY RUN: Would update CLAUDE.md")
                    return True
                
                with open(self.claude_md_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.log("Updated CLAUDE.md with detected changes")
                return True
            else:
                self.log("No significant changes detected for CLAUDE.md update", "VERBOSE")
                return False
        
        except Exception as e:
            self.log(f"Error updating CLAUDE.md: {e}", "ERROR")
            return False
    
    def _update_cli_parameters_section(self, content: str, new_params: List[Dict[str, str]]) -> str:
        """Update the Key CLI Parameters section with new parameters."""
        if not new_params:
            return content
        
        # Find the CLI parameters section
        cli_section_pattern = r'(### Key CLI Parameters\n\n)(.*?)(\n##|\n### |$)'
        match = re.search(cli_section_pattern, content, re.DOTALL)
        
        if not match:
            return content
        
        existing_params = match.group(2)
        
        # Add new parameters
        for param_info in new_params:
            param_line = f"- `--{param_info['param']}` - {param_info['help']}\n"
            if param_line not in existing_params:
                existing_params += param_line
        
        # Replace the section
        new_section = match.group(1) + existing_params + match.group(3)
        content = content[:match.start()] + new_section + content[match.end():]
        
        return content
    
    def _update_critical_components(self, content: str, new_files: List[str], 
                                   analysis: Dict[str, any]) -> str:
        """Update critical components section with new significant files."""
        significant_files = []
        
        for file_path in new_files:
            if any(dir_name in file_path for dir_name in ['model/', 'ensemble/', 'opt/', 'scripts/']):
                # Check if it has significant classes or functions
                file_classes = [c for c in analysis["new_classes"] if c["file"] == file_path]
                file_functions = [f for f in analysis["new_functions"] if f["file"] == file_path]
                
                if file_classes or len(file_functions) > 2:  # Significant if has classes or many functions
                    significant_files.append({
                        "path": file_path,
                        "classes": file_classes,
                        "functions": file_functions
                    })
        
        if not significant_files:
            return content
        
        # Add to appropriate subsection based on directory
        for file_info in significant_files:
            file_path = file_info["path"]
            
            if file_path.startswith('scripts/'):
                # Add to a development tools section or create one
                dev_section_pattern = r'(### Development Commands\n)(.*?)(\n###|\n##|$)'
                match = re.search(dev_section_pattern, content, re.DOTALL)
                
                if match:
                    description = f"- `{file_path}` - **NEW**: Automated development utility"
                    if file_info["classes"]:
                        description += f" with {', '.join([c['name'] for c in file_info['classes']])}"
                    description += "\n"
                    
                    updated_section = match.group(1) + match.group(2) + description + match.group(3)
                    content = content[:match.start()] + updated_section + content[match.end():]
        
        return content
    
    def _update_known_issues_section(self, content: str, status: Dict[str, List[str]], 
                                    date: str) -> str:
        """Update known issues section if fixes are detected."""
        # This is a placeholder - in practice, you'd analyze commit messages 
        # and file changes to detect specific bug fixes
        _ = status, date  # Mark as intentionally unused
        return content
    
    def _update_completed_testing_section(self, content: str, change_types: Set[str], 
                                        analysis: Dict[str, any]) -> str:
        """Update completed testing section with new experimental results."""
        # This would be expanded based on specific experimental frameworks
        _ = change_types, analysis  # Mark as intentionally unused
        return content
    
    def generate_commit_message(self, status: Dict[str, List[str]], 
                               change_types: Set[str], analysis: Dict[str, any]) -> str:
        """Generate an intelligent commit message based on detected changes."""
        
        all_files = status["modified"] + status["untracked"] + status["staged"]
        num_files = len(all_files)
        
        # Determine primary action
        if "untracked" in status and status["untracked"]:
            if len(status["untracked"]) > len(status["modified"]):
                action = "Add"
            else:
                action = "Update"
        elif status["modified"]:
            action = "Update"
        else:
            action = "Enhance"
        
        # Determine primary focus
        focus_map = {
            "algorithm": "algorithm implementation",
            "code": "core functionality", 
            "documentation": "documentation",
            "configuration": "configuration",
            "analysis": "analysis tools",
            "evaluation": "evaluation framework",
            "data_pipeline": "data pipeline",
            "artifacts": "output artifacts"
        }
        
        primary_focus = None
        for change_type in ["algorithm", "code", "analysis", "documentation", "configuration"]:
            if change_type in change_types:
                primary_focus = focus_map[change_type]
                break
        
        if not primary_focus:
            primary_focus = "system components"
        
        # Build commit message
        if analysis.get("new_cli_params"):
            message = f"{action} {primary_focus} with new CLI parameters"
        elif "scripts" in str(all_files).lower():
            message = f"{action} development automation tools"
        elif len(change_types) > 2:
            message = f"{action} multiple system components"
        else:
            message = f"{action} {primary_focus}"
        
        # Add detail if meaningful
        if num_files == 1:
            filename = all_files[0]
            if not filename.endswith('.md'):
                message += f": {Path(filename).stem}"
        elif 2 <= num_files <= 3:
            message += f" ({num_files} files)"
        elif num_files > 3:
            message += f" across {num_files} files"
        
        return message
    
    def perform_git_operations(self, custom_message: Optional[str] = None, 
                              push: bool = True) -> bool:
        """Stage, commit, and optionally push changes."""
        
        # Get current status
        status = self.get_git_status()
        
        if not any(status.values()):
            self.log("No changes to commit")
            return True
        
        # Analyze changes
        change_types = self.detect_change_types(status)
        all_files = status["modified"] + status["untracked"]
        analysis = self.analyze_code_changes(all_files)
        
        self.log(f"Detected changes: {', '.join(sorted(change_types))}")
        
        # Stage all changes
        self.log("Staging all changes...")
        success, _ = self.run_command(["git", "add", "."])
        if not success:
            return False
        
        # Generate commit message
        if custom_message:
            commit_msg = custom_message
        else:
            commit_msg = self.generate_commit_message(status, change_types, analysis)
        
        # Add standard footer
        footer = "\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
        full_message = commit_msg + footer
        
        # Commit changes
        self.log(f"Committing with message: {commit_msg}")
        success, _ = self.run_command([
            "git", "commit", "-m", full_message
        ])
        
        if not success:
            return False
        
        # Push to remote
        if push:
            self.log("Pushing to remote repository...")
            success, _ = self.run_command(["git", "push"])
            if not success:
                self.log("Push failed, but commit succeeded", "WARNING")
                return False
        
        return True
    
    def compact_session(self) -> bool:
        """Execute /compact command if available."""
        # This would interact with Claude's session compacting if available
        # For now, just log that it would be done
        self.log("Session compacting not implemented in this version", "WARNING")
        return True
    
    def run_wrapup(self, custom_message: Optional[str] = None, no_push: bool = False,
                   no_compact: bool = False, no_claude_update: bool = False) -> bool:
        """Run the complete wrap-up process."""
        
        self.log("Starting auto wrap-up process...")
        
        # Check if we're in a git repository
        if not (self.root_dir / ".git").exists():
            self.log("Not in a git repository", "ERROR")
            return False
        
        # Get current status for analysis
        status = self.get_git_status()
        
        if not any(status.values()):
            self.log("No changes detected - nothing to wrap up")
            return True
        
        # Analyze changes
        change_types = self.detect_change_types(status) 
        all_files = status["modified"] + status["untracked"]
        analysis = self.analyze_code_changes(all_files)
        
        self.log(f"Files to process: {len(all_files)}")
        self.log(f"Change types: {', '.join(sorted(change_types))}", "VERBOSE")
        
        success = True
        
        # Update CLAUDE.md if requested
        if not no_claude_update:
            self.log("Updating CLAUDE.md...")
            claude_updated = self.update_claude_md(status, change_types, analysis)
            if claude_updated:
                # Re-analyze status after CLAUDE.md update
                status = self.get_git_status()
        
        # Compact session if requested
        if not no_compact:
            self.log("Compacting session...")
            success = self.compact_session() and success
        
        # Perform git operations
        self.log("Performing git operations...")
        success = self.perform_git_operations(
            custom_message=custom_message,
            push=not no_push
        ) and success
        
        if success:
            self.log("✅ Auto wrap-up completed successfully!")
        else:
            self.log("❌ Auto wrap-up completed with some errors", "WARNING")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Auto wrap-up script for XGBoost Ensemble Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--message", 
        type=str, 
        help="Custom commit message (otherwise auto-generated)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--no-push", 
        action="store_true",
        help="Don't push to remote repository"
    )
    parser.add_argument(
        "--no-compact", 
        action="store_true",
        help="Skip session compacting"  
    )
    parser.add_argument(
        "--no-claude-update", 
        action="store_true",
        help="Skip CLAUDE.md updates"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize the manager
    manager = AutoWrapupManager(
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Run the wrap-up process
    success = manager.run_wrapup(
        custom_message=args.message,
        no_push=args.no_push,
        no_compact=args.no_compact,
        no_claude_update=args.no_claude_update
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()