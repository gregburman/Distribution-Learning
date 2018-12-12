FROM funkey/gunpowder
LABEL maintainer gburman@ini.uzh.ch

# install graph tools
RUN printf "deb http://downloads.skewed.de/apt/xenial xenial universe" >> /etc/apt/sources.list && \
	printf "deb-src http://downloads.skewed.de/apt/xenial xenial universe" >> /etc/apt/sources.list

RUN \
	apt-get -y update && \
	apt-get -y install python-graph-tool --allow-unauthenticated && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

#RUN apt-get source python-graph-tool
#RUN apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25

# install skelerator
RUN git clone https://github.com/nilsec/skelerator.git
WORKDIR skelerator
RUN python setup.py install

WORKDIR /distribution-learning
COPY . /distribution-learning

# install python library dependencies
RUN pip install -r requirements.txt

EXPOSE 80

# CMD ["python", "train.py"]
