# Navigate to your project folder
cd /path/to/your/fraud-detection-project

# Make individual scripts executable
chmod +x scripts/deploy_to_aws.sh
chmod +x scripts/cleanup_aws.sh

# Or make all .sh files executable at once
chmod +x scripts/*.sh

# Verify they're executable (look for 'x' in permissions)
ls -l scripts/*.sh
```

You should see output like:
```
-rwxr-xr-x  1 user  staff  12345 Nov 27 10:00 deploy_to_aws.sh
-rwxr-xr-x  1 user  staff   5678 Nov 27 10:00 cleanup_aws.sh