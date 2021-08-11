pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"koushik.nov01@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml