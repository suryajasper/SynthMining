import mongoose, { isObjectIdOrHexString } from 'mongoose';
import { ObjectId } from 'mongoose';

const ProjectSchema: mongoose.Schema = new mongoose.Schema({
  name: String,
  description: String,

  patronId: isObjectIdOrHexString,
  tags: [String],
});

export default mongoose.model('Project', ProjectSchema);