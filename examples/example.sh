curl -O https://nrvis.com/download/data/dynamic/fb-forum.zip > /dev/null 2>&1

unzip fb-forum.zip -d ./

rm fb-forum.zip

rm -rf readme.html

mv fb-forum.edges fb