# Git Command

## Checkout Remote Git Branch

The answer has been split depending on whether there is one remote repository configured or multiple. The reason for this is that for the single remote case, some of the commands can be simplified as there is less ambiguity.

In both cases, start by fetching from the remote repository to make sure you have all the latest changes downloaded.

```
$ git fetch
```

This will fetch all of the remote branches for you. You can see the branches available for checkout with:

```
$ git branch -v -a
```

In the case where multiple remote repositories exist, the remote repository needs to be explicitly named.

As before, start by fetching the latest remote changes:

```
$ git fetch origin
```

This will fetch all of the remote branches for you. You can see the branches available for checkout with:

```
$ git branch -v -a
```

With the remote branches in hand, you now need to check out the branch you are interested in with -c to create a new local branch:

```
$ git switch -c test origin/test
```

For more information about using git switch:

```
$ man git-switch
```

## Git Merge Master into Branch

To re-synchronise a branch with updates that have been made to the main branch on the repository, first ensure the local main branch has been updated using a checkout and pull for the main branch. Then checkout the branch of interest and merge from the updated local main. We can then push the merges back to the remote repository's version of the branch. The commits are those that were committed by others to the remote repository's main branch.

```
$ git checkout main
$ git pull
$ git checkout validator
$ git merge main
$ git push
```

Notice that we could skip the first two lines and change the merge to merge origin/main to also effect a merge from the remote main into the current branch. This will not then update the local copy of main.

```
$ git checkout validator
$ git merge origin/main
$ git push
```
