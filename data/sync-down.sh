rsync -rav -e ssh \
	--include '*/' \
	climbach@login.bswift.umd.edu:/data/bswift-1/Pessoa_Lab/eCON/ApprRetrSeg/processed/ \
	/home/climbach/approach-retreat/data/

