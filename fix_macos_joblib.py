#!/usr/bin/env python3
"""
Quick fix for macOS joblib subprocess issues.
This script sets environment variables to disable parallel processing in joblib.
Run this before executing any notebook to avoid subprocess errors.
"""

import os

# Set environment variables to disable joblib parallel processing
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['JOBLIB_START_METHOD'] = 'threading'
os.environ['SKLEARN_DISABLE_THREADING'] = '1'

print("macOS joblib fix applied!")
print("Environment variables set:")
print(f"  LOKY_MAX_CPU_COUNT = {os.environ.get('LOKY_MAX_CPU_COUNT')}")
print(f"  JOBLIB_START_METHOD = {os.environ.get('JOBLIB_START_METHOD')}")
print(f"  SKLEARN_DISABLE_THREADING = {os.environ.get('SKLEARN_DISABLE_THREADING')}")
print("\nYou can now run your notebook without subprocess errors.")
print("Note: This will use single-threaded processing (slower but stable).")