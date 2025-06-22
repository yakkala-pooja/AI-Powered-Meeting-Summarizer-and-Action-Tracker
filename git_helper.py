#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
from typing import List, Optional, Tuple

def run_command(command: List[str]) -> Tuple[str, str, int]:
    """Run a shell command and return stdout, stderr, and return code."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode

def git_status() -> None:
    """Show the current git status."""
    stdout, stderr, code = run_command(['git', 'status'])
    if code != 0:
        print(f"Error: {stderr}")
        sys.exit(1)
    print(stdout)

def git_add_all() -> None:
    """Add all changes to git."""
    stdout, stderr, code = run_command(['git', 'add', '.'])
    if code != 0:
        print(f"Error adding files: {stderr}")
        sys.exit(1)
    print("All changes added to git.")
    git_status()

def git_add_files(files: List[str]) -> None:
    """Add specific files to git."""
    command = ['git', 'add'] + files
    stdout, stderr, code = run_command(command)
    if code != 0:
        print(f"Error adding files: {stderr}")
        sys.exit(1)
    print(f"Added {len(files)} files to git.")
    git_status()

def git_commit(message: str) -> None:
    """Commit changes with the given message."""
    stdout, stderr, code = run_command(['git', 'commit', '-m', message])
    if code != 0:
        print(f"Error committing changes: {stderr}")
        sys.exit(1)
    print(stdout)

def git_push() -> None:
    """Push changes to remote repository."""
    stdout, stderr, code = run_command(['git', 'push'])
    if code != 0:
        print(f"Error pushing changes: {stderr}")
        sys.exit(1)
    print(stdout)

def git_pull() -> None:
    """Pull changes from remote repository."""
    stdout, stderr, code = run_command(['git', 'pull'])
    if code != 0:
        print(f"Error pulling changes: {stderr}")
        sys.exit(1)
    print(stdout)

def list_untracked_files() -> None:
    """List files that are not tracked by git."""
    stdout, stderr, code = run_command(['git', 'ls-files', '--others', '--exclude-standard'])
    if code != 0:
        print(f"Error listing untracked files: {stderr}")
        sys.exit(1)
    
    files = stdout.strip().split('\n')
    if files == ['']:
        print("No untracked files found.")
        return
    
    print("Untracked files:")
    for file in files:
        print(f"  {file}")

def list_modified_files() -> None:
    """List files that have been modified."""
    stdout, stderr, code = run_command(['git', 'ls-files', '--modified'])
    if code != 0:
        print(f"Error listing modified files: {stderr}")
        sys.exit(1)
    
    files = stdout.strip().split('\n')
    if files == ['']:
        print("No modified files found.")
        return
    
    print("Modified files:")
    for file in files:
        print(f"  {file}")

def git_clean(directories: bool = False) -> None:
    """Remove untracked files from the working tree."""
    command = ['git', 'clean', '-f']
    if directories:
        command.append('-d')
    
    stdout, stderr, code = run_command(command)
    if code != 0:
        print(f"Error cleaning untracked files: {stderr}")
        sys.exit(1)
    print(stdout)

def main() -> None:
    """Main function to parse arguments and execute git commands."""
    parser = argparse.ArgumentParser(description='Git Helper Script')
    
    subparsers = parser.add_subparsers(dest='command', help='Git command to run')
    
    # Status command
    subparsers.add_parser('status', help='Show git status')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add files to git')
    add_parser.add_argument('--all', '-a', action='store_true', help='Add all files')
    add_parser.add_argument('files', nargs='*', help='Files to add')
    
    # Commit command
    commit_parser = subparsers.add_parser('commit', help='Commit changes')
    commit_parser.add_argument('message', help='Commit message')
    
    # Push command
    subparsers.add_parser('push', help='Push changes to remote')
    
    # Pull command
    subparsers.add_parser('pull', help='Pull changes from remote')
    
    # List untracked files command
    subparsers.add_parser('untracked', help='List untracked files')
    
    # List modified files command
    subparsers.add_parser('modified', help='List modified files')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Remove untracked files')
    clean_parser.add_argument('--directories', '-d', action='store_true', help='Remove untracked directories too')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        git_status()
    elif args.command == 'add':
        if args.all:
            git_add_all()
        elif args.files:
            git_add_files(args.files)
        else:
            print("Error: Please specify files to add or use --all")
            sys.exit(1)
    elif args.command == 'commit':
        git_commit(args.message)
    elif args.command == 'push':
        git_push()
    elif args.command == 'pull':
        git_pull()
    elif args.command == 'untracked':
        list_untracked_files()
    elif args.command == 'modified':
        list_modified_files()
    elif args.command == 'clean':
        git_clean(args.directories)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 