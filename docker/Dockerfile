FROM funkey/gunpowder:v1.1
LABEL maintainer gburman@ini.uzh.ch

# RUN pip install --upgrade tensorflow-probability

# install graph tools
RUN printf "deb http://downloads.skewed.de/apt/xenial xenial universe" >> /etc/apt/sources.list && \
	printf "deb-src http://downloads.skewed.de/apt/xenial xenial universe" >> /etc/apt/sources.list

RUN \
	apt-get -y update && \
	apt-get -y install python-graph-tool --allow-unauthenticated && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN \
	apt-get -y update && \
	apt-get -y install libmlpack-dev --allow-unauthenticated

#RUN apt-get source python-graph-tool
#RUN apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25

RUN git clone https://github.com/funkey/mala.git
WORKDIR mala
RUN python setup.py install

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
