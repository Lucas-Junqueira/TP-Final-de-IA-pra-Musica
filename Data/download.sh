#!/usr/bin/env bash
set -e
mkdir -p data/raw && cd data/raw

# EMOPIA+ ("+" vira %2B na URL)
wget -O EMOPIA_plus.zip "https://zenodo.org/records/13122742/files/EMOPIA%2B.zip?download=1"
echo "821f6bb615053ac2e7171922198b58e2  EMOPIA_plus.zip" | md5sum -c
unzip -q EMOPIA_plus.zip -d emopia_plus

# Pop1k7
wget -O Pop1K7.zip "https://zenodo.org/records/13167761/files/Pop1K7.zip?download=1"
echo "a75ead698067502bcaaa332af4880f83  Pop1K7.zip" | md5sum -c
unzip -q Pop1K7.zip -d pop1k7

# VGMIDI (opcional, só se sobrar tempo)
# git clone https://github.com/lucasnfe/vgmidi.git