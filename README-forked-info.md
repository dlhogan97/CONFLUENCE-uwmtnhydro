# Instructions: Pulling Updates from Original Repository into a Fork

## 1. Check Existing Remotes

Run this to see current remotes:

    git remote -v

Typical output for a fork:

    origin  https://github.com/your-username/forked-repo.git (fetch)
    origin  https://github.com/your-username/forked-repo.git (push)

## 2. Add the Original Repository as "upstream"

If not already set, add the original repo:

    git remote add upstream https://github.com/original-owner/original-repo.git

Confirm the remotes:

    git remote -v

You should now see:

    origin    https://github.com/your-username/forked-repo.git (fetch)
    origin    https://github.com/your-username/forked-repo.git (push)
    upstream  https://github.com/original-owner/original-repo.git (fetch)
    upstream  https://github.com/original-owner/original-repo.git (push)

## 3. Pull Changes from the Original Repository

To merge changes from the upstream repo's main branch:

    git pull upstream main

(Replace `main` with `master` if applicable.)

## 4. Optional: Rebase Instead of Merge

For a cleaner commit history:

    git fetch upstream
    git rebase upstream/main

## 5. Optional: Push Updates to Your Fork

After updating your local repo, you can push to your forked GitHub repo:

    git push origin main

(Again, use your branch name if not `main`.)

## Notes

- Use `git branch` to confirm your current branch.
- Use a GitHub personal access token if prompted for credentials.
The default branch has been renamed!
main is now named hogan-main

If you have a local clone, you can update it by running the following commands.

    ```git branch -m main hogan-main
    git fetch origin
    git branch -u origin/hogan-main hogan-main
    git remote set-head origin -a```
