import json
from code.fetch_data import fetchFiles
json_files = fetchFiles()
fileptr= open("evalution.csv","a")
fileptr.write('created_at'+","+'text'+'\n')

for i in range(0,len(json_files)):
    for j in range(0,len(json_files[i])):
        fileptr.write(str(json_files[i][j]['created_at']) +","+json_files[i][j]['text'].encode("utf-8"))
        fileptr.write('\n')
        fileptr.write('\n')
        fileptr.write('\n')
        fileptr.write('\n')
        fileptr.write('\n')
        #str(json_files[i][j]['retweet_count'])+","+str(json_files[i][j]['favorite_count'])+","+str(json_files[i][j]['place'])+"\n")
        