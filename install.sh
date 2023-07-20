mkdir ./bin
cd ./bin
ln -sf ../exph.py ./
ln -sf ../Utilities/finite_any_Q_BSE.py ./
ln -sf ../Utilities/finite_any_Q_DFT.py ./
ln -sf ../Utilities/finite_Q_get_data.py ./
ln -sf ../Utilities/finite_uniform_Q_BSE.py ./
ln -sf ../Utilities/collect.py ./
ln -sf ../Utilities/dat2h5_gkk.py

chmod +x collect.py
chmod +x exph.py
chmod +x finite_any_Q_BSE.py
chmod +x finite_any_Q_DFT.py
chmod +x finite_Q_get_data.py
chmod +x finite_uniform_Q_BSE.py

echo 'link done!'

echo 'please add this' $(pwd) 'to your env variable'