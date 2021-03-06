#!/usr/bin/env bash
set -e

usage() {
  echo "Usage: $0 [-w <working directory>] -d <docker image> -e <environmental variable>  <command> [<args>...]";
  echo;
  echo "    -w  The working directory for <command>, defaults to current";
  echo "        directory.";
  echo;
  echo "    -d  Docker image to use to execute <command>. The docker container";
  echo "        will have";
  echo "            /groups";
  echo "            /nrs";
  echo "            /scratch";
  echo "        available.";
  echo;
  echo "    -p  Ports to expose, see 'man docker run'.";
  echo;
  echo "    -e  Optional environmental variable for docker image.";
}

# defaults

WORK_DIR=$(pwd)
DOCKER_IMAGE=''
PORTS=''
ENVI=''
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# parse command line args

num_args=0
while getopts ":w:d:p:e:" opt; do
  case "${opt}" in
    w)
        WORK_DIR=${OPTARG}
        ((num_args+=2))
        ;;
    d)
        DOCKER_IMAGE=${OPTARG}
        ((num_args+=2))
        ;;
    p)
        PORTS="-p ${OPTARG}"
        ((num_args+=2))
        ;;
    e)
        ENVI="-e ${OPTARG}"
        ((num_args+=2))
        ;;
    *)
        usage
        exit
        ;;
  esac
done

shift ${num_args}
COMMAND="$@"

if [ "$DOCKER_IMAGE" == "" ];
then \
  echo No docker image provided!
  echo
  usage
  exit
fi

if [ "${COMMAND}" == "" ];
then \
  echo No command provided!
  echo
  usage
  exit
fi

CONTAINER_NAME=${CONTAINER_NAME:-${USER}-$(date +%y-%m-%d_%H-%M-%S)-${RANDOM}}

# make sure we have the latest
nvidia-docker pull ${DOCKER_IMAGE}

teardown() {
  trap - SIGINT SIGTERM
  echo "run_docker: Stopping container ${CONTAINER_NAME}, killing after 5s..."
  docker stop -t5 ${CONTAINER_NAME}
  echo "run_docker: Container ${CONTAINER_NAME} stopped."
}

trap teardown SIGINT SIGTERM

echo "Running docker as user:group `id -u $USER`:`id -g $USER`"

echo "nvidia-docker \
  run \
  --rm \
  --cgroup-parent=$(cat /proc/self/cpuset) \
  --name ${CONTAINER_NAME} \
  -e HOME="$HOME" \
  -u `id -u $USER`:`id -g $USER` \
  -v  ${WORK_DIR} \
  -w ${WORK_DIR} \
  ${ENVI} \
  ${PORTS} \
  ${DOCKER_IMAGE} \
  ${COMMAND} &"

# run nvidia-docker in background
export NV_GPU=$CUDA_VISIBLE_DEVICES
nvidia-docker \
  run \
  --rm \
  --cgroup-parent=$(cat /proc/self/cpuset) \
  --name ${CONTAINER_NAME} \
  -e HOME="$HOME" \
  -u `id -u $USER`:`id -g $USER` \
  -v  ${WORK_DIR} \
  -w ${WORK_DIR} \
  ${ENVI} \
  ${PORTS} \
  ${DOCKER_IMAGE} \
  ${COMMAND} &

wait $!