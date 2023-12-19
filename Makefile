test:
	python -m pytest -v --cov=protolearn --cov-report=term-missing protolearn/tests

get-banknote-data:
	wget https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip
	unzip banknote+authentication.zip
	rm banknote+authentication.zip
	mv data_banknote_authentication.txt data/banknote.txt


get-kc-data:
	wget https://geodacenter.github.io/data-and-lab/data/kingcounty.zip
	unzip -p kingcounty.zip kingcounty/kc_house_data.csv > data/kc_house_data.csv
	rm kingcounty.zip