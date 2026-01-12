"""
API Version Manager
===================
Manages model versions for deployment (v1, v2, rollback).
"""

import shutil
import os
import json
from datetime import datetime

MODELS_DIR = "models"
VERSIONS_DIR = "models/versions"
CURRENT_MODEL = "models/model.joblib"
VERSION_FILE = "models/version_info.json"


def init_versioning():
    """Initialize version directories."""
    os.makedirs(VERSIONS_DIR, exist_ok=True)
    if not os.path.exists(VERSION_FILE):
        save_version_info({"current": None, "history": []})
    print("✓ Version management initialized")


def save_version_info(info):
    """Save version information to JSON."""
    with open(VERSION_FILE, 'w') as f:
        json.dump(info, f, indent=2)


def load_version_info():
    """Load version information from JSON."""
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r') as f:
            return json.load(f)
    return {"current": None, "history": []}


def create_version(version_name, description=""):
    """
    Create a new version from the current model.
    
    Args:
        version_name: e.g., "v1", "v2"
        description: Description of this version
    """
    init_versioning()
    
    if not os.path.exists(CURRENT_MODEL):
        print("❌ No model found at", CURRENT_MODEL)
        return False
    
    # Copy current model to version directory
    version_path = os.path.join(VERSIONS_DIR, f"model_{version_name}.joblib")
    shutil.copy2(CURRENT_MODEL, version_path)
    
    # Update version info
    info = load_version_info()
    version_entry = {
        "version": version_name,
        "path": version_path,
        "description": description,
        "created_at": datetime.now().isoformat()
    }
    info["history"].append(version_entry)
    info["current"] = version_name
    save_version_info(info)
    
    print(f"✅ Created version: {version_name}")
    print(f"   Path: {version_path}")
    print(f"   Description: {description}")
    return True


def deploy_version(version_name):
    """
    Deploy a specific version as the current model.
    
    Args:
        version_name: Version to deploy (e.g., "v1", "v2")
    """
    version_path = os.path.join(VERSIONS_DIR, f"model_{version_name}.joblib")
    
    if not os.path.exists(version_path):
        print(f"❌ Version {version_name} not found at {version_path}")
        return False
    
    # Copy version to current model
    shutil.copy2(version_path, CURRENT_MODEL)
    
    # Update version info
    info = load_version_info()
    info["current"] = version_name
    save_version_info(info)
    
    print(f"✅ Deployed version: {version_name}")
    return True


def rollback(to_version=None):
    """
    Rollback to a previous version.
    
    Args:
        to_version: Specific version to rollback to. If None, rollback to previous.
    """
    info = load_version_info()
    
    if not info["history"]:
        print("❌ No versions available for rollback")
        return False
    
    if to_version:
        # Rollback to specific version
        target = to_version
    else:
        # Rollback to previous version
        current = info["current"]
        versions = [v["version"] for v in info["history"]]
        if current in versions:
            idx = versions.index(current)
            if idx > 0:
                target = versions[idx - 1]
            else:
                print("❌ Already at oldest version")
                return False
        else:
            target = versions[-1]
    
    print(f"🔄 Rolling back from {info['current']} to {target}...")
    return deploy_version(target)


def get_current_version():
    """Get the currently deployed version."""
    info = load_version_info()
    return info.get("current", "unknown")


def list_versions():
    """List all available versions."""
    info = load_version_info()
    
    print("\n" + "="*50)
    print("📋 AVAILABLE VERSIONS")
    print("="*50)
    
    if not info["history"]:
        print("No versions created yet.")
        return
    
    for v in info["history"]:
        marker = "→ " if v["version"] == info["current"] else "  "
        print(f"{marker}{v['version']}: {v['description']} ({v['created_at'][:10]})")
    
    print("="*50)
    print(f"Current: {info['current']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Version Manager")
    parser.add_argument("action", choices=["create", "deploy", "rollback", "list", "current"])
    parser.add_argument("--version", "-v", type=str, help="Version name")
    parser.add_argument("--description", "-d", type=str, default="", help="Version description")
    
    args = parser.parse_args()
    
    if args.action == "create":
        if not args.version:
            print("❌ Please specify version name with --version")
        else:
            create_version(args.version, args.description)
    elif args.action == "deploy":
        if not args.version:
            print("❌ Please specify version name with --version")
        else:
            deploy_version(args.version)
    elif args.action == "rollback":
        rollback(args.version)
    elif args.action == "list":
        list_versions()
    elif args.action == "current":
        print(f"Current version: {get_current_version()}")
