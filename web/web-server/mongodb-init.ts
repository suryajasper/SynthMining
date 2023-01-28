import { MongoClient, ServerApiVersion } from 'mongodb';
import mongoose from 'mongoose';
import creds from '../../secret.json';

const uri: string = `mongodb+srv://${creds.mdb.username}:${creds.mdb.password}@sm-main-cluster.rg540zq.mongodb.net/?retryWrites=true&w=majority`;
mongoose.connect(uri);

const db: mongoose.Connection = mongoose.connection;

export default db;