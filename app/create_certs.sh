##Create self signed SSL Certificate for more secure access

openssl genpkey -algorithm RSA -out localhost.key -aes256

##Create cert signing request, USE localhost when prompted for Common Name (CN)

openssl req -new -key localhost.ket -out localhost.csr

##Self Signed Certificate
openssl x509 -req -days 365 -in localhost.csr -signkey localhost.key -out localhost.crt



## for linux , copy cert to  usr/local/share/ca-certificates dir

sudo cp localhost.crt /usr/local/share/ca-certificates

##for Windows, and mac see https://betterstack.com/community/questions/getting-chrome-to-accept-self-signed-localhost-certificate/


