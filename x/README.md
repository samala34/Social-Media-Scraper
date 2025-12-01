Add TWITTER_BEARER_TOKEN in .env file inside backend folder
Open Backend and frontend in two seperate terminals
Backend - pip install -r requirements.txt
          python app.py
Frontend/myapp - npm install
          npm start

There are 2 backend files
1. app.py
          This backend script will take a natural query from user and parse the query and find entities (username, number of tweets, topic.....)
             in here this script have similarity search, it will find the similarity score between the user query and tweets and then return top tweets based on score

3. app2.py
          This backend script will take a natural query from user and parse the query and find entities (username, number of tweets, topic.....)
             in this the scripts,it directly take the topic as hashtag and pass it with API and return data
