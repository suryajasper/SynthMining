import React from 'react';
import { fetchRequest, randId } from '../utils/utils';
import { icons } from './icons';

type ImageFile = {
  file: File;
  id: string;
}

interface ImageUploadAttrs {
  children: JSX.Element | JSX.Element[];

  uid: string | undefined;
  projectId: string | undefined;
  active: boolean;

  status: (res: any) => void;
}

interface ImageUploadState {
  dragOver: boolean;
  uploading: boolean;
  curr: number;
  total: number;
}

export default class ImageUpload 
  extends React.Component<ImageUploadAttrs, ImageUploadState> {
  
  constructor(props: ImageUploadAttrs) {
    super(props);

    this.state = {
      dragOver: false,
      uploading: false,
      curr: 0,
      total: 1,
    }
  }

  uploadToServer(imgBatch : ImageFile[]) {
    if (!this.props.uid || !this.props.projectId) return;

    const formdata : FormData = new FormData();
    
    for (let i = 0; i < imgBatch.length; i++) {
      formdata.append(`${i}`, imgBatch[i].file);
      formdata.append('img_ids[]', imgBatch[i].id);
    }
    
    formdata.append('uid', this.props.uid);

    formdata.append('batch_size', imgBatch.length.toString());
    
    return new Promise((res, rej) => {
      var xhr = new XMLHttpRequest();
      xhr.open("POST", "http://localhost:2003/uploadImages", true);
      xhr.onload = function() {
        if (this.status === 200) res('success');
        else rej();
      };
      xhr.send(formdata);
    });
  }

  async getImageIds(imgNames: string[]) : Promise<string[]> {
    const { imageIds } = await fetchRequest<{imageIds: string[]}>('/addImages', {
      method: 'POST',
      body: {
        uid: this.props.uid,
        projectId: this.props.projectId,
        imgNames,
      }
    });

    return imageIds;
  }

  // updateUploadStateSynch(update: { uploading?: boolean, curr?: number, total?: number }) {
  //   return new Promise(resolve => {
  //     this.setState(prevState => ({
  //       ...prevState,
  //       ...update,
  //     }), () => { resolve(null); });
  //   })
  // }

  async uploadImages(files : File[]) {
    const BATCH_SIZE: number = 1;

    this.setState({
      uploading: true,
      curr: 0,
      total: files.length,
    });

    let fileNames = files.map(
      file => file.name.split('.')[0]
    );

    let imageIds = await this.getImageIds(fileNames);

    let fileCopy: ImageFile[] = files.map((file, i) => {
      return { file, id: imageIds[i], };
    });
    
    while (fileCopy.length > 0) {
      let batch: ImageFile[] = fileCopy.splice(0, BATCH_SIZE);
      
      await this.uploadToServer(batch);
      
      this.setState(prevState => ({
        curr: prevState.curr + batch.length,
      }));
    }

    await this.setState({
      curr: 0,
      total: 1,
    })

    this.props.status('success');
  }

  containsFiles(event: React.DragEvent) : boolean {
    if (event.dataTransfer?.types) {
      for (var i = 0; i < event.dataTransfer.types.length; i++) {
        if (event.dataTransfer.types[i] == "Files") {
          return true;
        }
      }
    }
    
    return false;
  }

  displayUploadInstructions() : JSX.Element {
    return (
      <>
        <div className='upload-header'>{
          this.state.uploading ? 
            `Uploading Images... ${this.state.curr} / ${this.state.total}` :
            'Drag / Click to Upload Files'
        }</div>

        <div className='upload-svg-container'>{ icons.upload }</div>
      </>
    );
  }

  displayProgressBar() : JSX.Element {
    return (
      <div className='progress-bar' style={{
        width: `${this.state.curr / this.state.total * 100}%`,
      }} />
    );
  }

  render() {
    return (
      <>
        <div 
          className={
            [
              'upload-container',
              this.props.active ? 'upload-foreground' : 'upload-background',
              this.state.dragOver ? 'upload-dragover' : '',
            ]
              .join(' ')
          }

          onDragOver={ e => {
            e.preventDefault();
  
            if (this.containsFiles(e))
              this.setState({dragOver: true});
          } }

          onDragLeave={ e => {
            e.preventDefault();
            this.setState({dragOver: false});
          }}

          onDrop={ e => {
            e.preventDefault();
  
            this.setState({dragOver: false});
  
            if (this.containsFiles(e) && e.dataTransfer)
              this.uploadImages(Array.from(e.dataTransfer.files));
          } }

          onClick={ () => {
            const fileInputEl : HTMLElement = document.querySelector('#imageIn') as HTMLElement;
            fileInputEl.click();
          } }
        >
          <>
            {this.displayProgressBar()}

            <div className='upload-content'>{
              this.props.active ?
                this.displayUploadInstructions() :
                this.props.children
            }</div>
          </>
        </div>

        <input 
          type="file"
          accept='image/png, image/jpeg'
          id="imageIn"
          name="imgFile"
          multiple 
          hidden
          onChange={ async e => {
            await this.uploadImages(Array.from(e.target.files));
          } }
        />
      </>
    );
  }
}