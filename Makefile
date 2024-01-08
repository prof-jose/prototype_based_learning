# Run tests
test:
	python -m pytest -v --cov=protolearn --cov-report=term-missing protolearn/tests


# Downlaod the necessary data
get-data: get-banknote-data get-kc-data


# Download the banknote data
get-banknote-data:
	wget https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip
	unzip banknote+authentication.zip
	rm banknote+authentication.zip
	mv data_banknote_authentication.txt data/banknote.txt


# Download the King County data
get-kc-data:
	wget https://geodacenter.github.io/data-and-lab/data/kingcounty.zip
	unzip -p kingcounty.zip kingcounty/kc_house_data.csv > data/kc_house_data.csv
	rm kingcounty.zip