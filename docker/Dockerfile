FROM funkey/gunpowder:v0.3.2
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

# install skelerator - use gregburman repo for unmerged changes, and checkout branch if necessary
RUN git clone https://github.com/nilsec/skelerator.git
# RUN git clone https://github.com/gregburman/skelerator.git
WORKDIR skelerator
# RUN git checkout noise_strength
RUN python setup.py install

WORKDIR /distribution-learning
COPY . /distribution-learning

# install python library dependencies
# RUN pip install -r requirements.txt

EXPOSE 80

CMD ["echo", "complete"]
