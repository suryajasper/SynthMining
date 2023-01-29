import mongoose, { isObjectIdOrHexString } from 'mongoose';
import { ObjectId } from 'mongodb';

const ImageSchema: mongoose.Schema = new mongoose.Schema({
  filePath: String,
  authorId: ObjectId,
  validated: Boolean,
})

const ProjectSchema: mongoose.Schema = new mongoose.Schema({
  name: String,
  description: String,

  patronId: ObjectId,
  tags: [String],

  imgPaths: [ImageSchema],
});

export default mongoose.model('Project', ProjectSchema);