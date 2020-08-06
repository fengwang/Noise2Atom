#
# sending message to telegram
#
import os

private_key = None # your private key
private_id = None # your private id


def send_message( message ):
    global private_key
    global private_id

    if private_key is None or private_id is None:
        return 0

    command = f'/usr/bin/curl -s -X POST https://api.telegram.org/{private_key}/sendMessage -d chat_id={private_id} -d text="{message}"'
    os.system( command )
    print( '*'*10 )

def send_photo( photo_path ):
    global private_key
    global private_id

    if private_key is None or private_id is None:
        return 0

    command = f'/usr/bin/curl -s -X POST https://api.telegram.org/{private_key}/sendPhoto -F chat_id={private_id} -F photo="@{photo_path}"'
    os.system( command )
    print( '*'*10 )

if __name__ == '__main__':
    send_message( 'Test message: message from python' )
    #send_photo( './10002073854_53dc1acd33_o.jpg' )

