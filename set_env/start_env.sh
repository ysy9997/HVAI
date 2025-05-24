# install apt packages
apt update


# install pip packages
python -m pip install --upgrade pip
pip install -r requirements.txt


# install vscode
cd /
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz
./code tunnel --accept-server-license-terms --disable-telemetry
