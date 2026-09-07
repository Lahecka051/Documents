# SSH Connection Config

## Host Alias

```sshconfig
Host ikm
    HostName 10.131.133.80
    User ikm
```

## Connection

```powershell
ssh ikm
```

## Verification

- SSH port: `22`
- Host reachable: `10.131.133.80`
- TCP connection test: passed
- Password authentication: passed
- Test command result: `ssh-ok`

## Notes

- The password was verified during connection testing, but it is not stored in this file.
- Keep the password in a secure password manager or use SSH key authentication for regular access.

# Project Work Rules

## Conversation And Change Logging

- For every project task, refer to this `config.md` before acting.
- If new requirements, operating rules, environment details, or important changes appear during work, record them in this file.
- Treat this file as required project context for future conversations.

## Server Usage

- Use only GPUs that satisfy the project idle-GPU eligibility rule.
- Never use or share a GPU occupied by another user. An occupied GPU is excluded
  regardless of its reported utilization, memory use, owner, or permission to share.
- Monitor GPU usage with:

```bash
gpustat -i 1
```

- Because server storage is limited, upload dataset-like files to NAS instead of keeping them on the server.
- NAS path on the server:

```bash
/NasData
```

## Python Environment

- Use `conda` as the default virtual environment manager.
- Server Miniconda is installed at:

```bash
/data2/ikm/miniconda3
```

- Installed conda version on the server: `conda 26.3.2`
- `conda init bash` has been applied for the server user `ikm`.
- `base` auto activation is disabled.
- In an interactive SSH shell, use conda normally:

```bash
conda --version
conda create -n <env-name> python=3.11
conda activate <env-name>
```

- In non-interactive SSH commands or scripts, source conda first:

```bash
source /data2/ikm/miniconda3/etc/profile.d/conda.sh
conda activate <env-name>
```

## Project Conda Environment

- Server environment name: `lkm`
- Environment path:

```bash
/data2/ikm/miniconda3/envs/lkm
```

- Verified Python version: `Python 3.10.20`
- Verified Python executable:

```bash
/data2/ikm/miniconda3/envs/lkm/bin/python
```

- Activation command:

```bash
source /data2/ikm/miniconda3/etc/profile.d/conda.sh
conda activate lkm
```
