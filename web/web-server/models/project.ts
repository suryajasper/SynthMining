import mongoose from 'mongoose';
import { ObjectId } from 'mongodb';

const TagSchema: mongoose.Schema = new mongoose.Schema({
  name: String,
  description: String,

  projectId: String,

  goalQty: {
    type: Number,
    required: true,
    validate: {
      validator: Number.isInteger,
      message: '{VALUE} is not an integer value'
    },
  },
})

const ImageSchema: mongoose.Schema = new mongoose.Schema({
  name: String,

  projectId: String,
  authorId: String,

  tags: [String],
  validated: Boolean,
})

const ProjectSchema: mongoose.Schema = new mongoose.Schema({
  name: String,
  description: String,
  keywords: [String],
  
  patronId: ObjectId,
});

const TagModel = mongoose.model('Tag', TagSchema);
const ImageModel = mongoose.model('Image', ImageSchema);
const ProjectModel = mongoose.model('Project', ProjectSchema);

export { TagModel, ImageModel, ProjectModel };