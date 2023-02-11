
export interface ImageAttrs {
  _id: string,
  name: string,

  projectId: string,
  authorId: string,

  tags: [string],
  validated: boolean,

  src: string,
}

export interface TagAttrs {
  _id: string,

  name: string,
  description: string,
  goalQty: number,

  projectId: string,
}

export interface ProjectBaseAttrs {
  _id: string,
  name: string,
  description: string,

  patronId: string,
  keywords: string[],
}

export interface ProjectAttrs extends ProjectBaseAttrs {
  images: ImageAttrs[],
  tags: TagAttrs[],
}