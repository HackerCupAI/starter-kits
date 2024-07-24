from huggingface_hub import snapshot_download

# Currently the dataset seems to be corrupted and for me this fails eventually with 
# OSError: Consistency check failed: file should be of size 42254566 but has size 42711476 (claw.in).
#
# But that's fine for now as enough files are downloaded before the failure.
snapshot_download(repo_id="hackercupai/hackercup", repo_type="dataset", allow_patterns=["*.in", "*.out", "*.txt"], local_dir="dataset")
