[[ $_ != $0 ]] \
    || { echo 1>&2 "source (.) this script with Bash."; exit 2; } 
(
    cd "$(dirname "$BASH_SOURCE")"
    [ -d .venv ] || {
        virtualenv .venv
        . .venv/bin/activate
        pip install -r requirements.txt
    }
)
. "$(dirname "$BASH_SOURCE")/.venv/bin/activate"