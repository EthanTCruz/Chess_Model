sudo apt install tmux neovim python3.11-venv git
python3 -m venv ./env
source ./env/bin/activate
export trainDataExists=False
export useSamplePgn=False
export trainModel=True
tmux new -s train
