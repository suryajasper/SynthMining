import http from 'http';
import app from './endpoints';

const server: http.Server = http.createServer(app);

const port : string | Number = process.env.PORT || 2002;

server.listen(port, () => {
  console.log(`Listening on http://127.0.0.1:${port}`);
})
