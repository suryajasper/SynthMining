import http from 'http';
import bodyParser from 'body-parser';
import connectMongoDB from './mongodb-init';
import app from './endpoints';

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Accept');
  next();
});
app.use(bodyParser.urlencoded({ extended: true }));

const server: http.Server = http.createServer(app);

const port : string | Number = process.env.PORT || 2002;

connectMongoDB()
  .then(db => {
    console.log('connected to mongodb successfully');

    server.listen(port, () => {
      console.log(`Listening on http://127.0.0.1:${port}`);
    })
  })
  .catch(console.error)

