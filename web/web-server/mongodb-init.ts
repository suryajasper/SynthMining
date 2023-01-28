import { MongoClient, ServerApiVersion } from 'mongodb';
import mongoose from 'mongoose';
import creds from '../../secret.json';

const uri: string = `mongodb+srv://${creds.mdb.username}:${creds.mdb.password}@sm-main-cluster.rg540zq.mongodb.net/?retryWrites=true&w=majority`;

export default function connectMongoDB() {
  return new Promise((res, rej) => {
    mongoose.connect(uri)
      .then(() => res(mongoose.connection))
      .catch(rej);
  })
}