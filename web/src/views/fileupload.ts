import m from 'mithril';
import { fetchRequest, randId } from '../utils/utils';

type ImageFile = {
  file: File;
  id: string;
}

interface ImageUploadAttrs {
  uid: string | undefined;
  projectId: string | undefined;
  active: boolean;
  status: (res: any) => void;
}

export default class ImageUpload implements m.ClassComponent<ImageUploadAttrs> {
  private status: (any) => void;

  private uid: string | undefined;
  private projectId: string | undefined;

  private upload: {
    dragOver: boolean;
    uploading: boolean;
    curr: number;
    total: number;
  }

  constructor(vnode : m.CVnode<ImageUploadAttrs>) {
    this.status = vnode.attrs.status;

    this.uid = vnode.attrs.uid;
    this.projectId = vnode.attrs.projectId;

    this.upload = {
      dragOver: false,
      uploading: false,
      curr: 0,
      total: 1,
    }
  }

  uploadToServer(imgBatch : ImageFile[]) {
    if (!this.uid || !this.projectId) return;

    const formdata : FormData = new FormData();
    
    for (let i = 0; i < imgBatch.length; i++) {
      formdata.append(`${i}`, imgBatch[i].file);
      formdata.append('img_ids[]', imgBatch[i].id);
    }
    
    formdata.append('uid', this.uid);

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
        uid: this.uid,
        projectId: this.projectId,
        imgNames,
      }
    });

    return imageIds;
  }

  async uploadImages(files : File[]) {
    const BATCH_SIZE: number = 1;

    this.upload.uploading = true;

    this.upload.curr = 0;
    this.upload.total = files.length;

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

      this.upload.curr += batch.length;

      m.redraw();
    }

    this.upload.curr = 0;
    this.upload.total = 1;

    this.status('success');
  }

  containsFiles(event: DragEvent) : boolean {
    if (event.dataTransfer?.types) {
      for (var i = 0; i < event.dataTransfer.types.length; i++) {
        if (event.dataTransfer.types[i] == "Files") {
          return true;
        }
      }
    }
    
    return false;
  }

  displayUploadInstructions() : m.Children {
    return [
      m('div.upload-header', this.upload.uploading ? 
        `Uploading Images... ${this.upload.curr} / ${this.upload.total}` :
        'Drag / Click to Upload Files'
      ),
      m('div.upload-svg-container', 
        m('svg', {
          viewBox:"0 0 24 24",
        }, m('path', {
          d: "M11 20H6.5Q4.22 20 2.61 18.43 1 16.85 1 14.58 1 12.63 2.17 11.1 3.35 9.57 5.25 9.15 5.88 6.85 7.75 5.43 9.63 4 12 4 14.93 4 16.96 6.04 19 8.07 19 11 20.73 11.2 21.86 12.5 23 13.78 23 15.5 23 17.38 21.69 18.69 20.38 20 18.5 20H13V12.85L14.6 14.4L16 13L12 9L8 13L9.4 14.4L11 12.85Z",
        }))
      ),
    ];
  }

  displayProgressBar() : m.Children {
    return m('div.progress-bar', {
      style: {
        width: `${this.upload.curr / this.upload.total * 100}%`,
      }
    });
  }

  view(vnode : m.CVnode<ImageUploadAttrs>) {
    return [
      m('div.upload-container', {
        class: [
          vnode.attrs.active ? 'upload-foreground' : 'upload-background',
          this.upload.dragOver ? 'upload-dragover' : '',
        ].join(' '),

        ondragover: (e: DragEvent) => {
          e.preventDefault();

          if (this.containsFiles(e))
            this.upload.dragOver = true;
        },
        ondragleave: (e : DragEvent) => {
          e.preventDefault();
          this.upload.dragOver = false;
        },
        ondrop: (e: DragEvent) => {
          e.preventDefault();

          this.upload.dragOver = false;

          if (this.containsFiles(e) && e.dataTransfer)
            this.uploadImages(Array.from(e.dataTransfer.files));
        },
        onclick: () => {
          const fileInputEl : HTMLElement = document.querySelector('#imageIn') as HTMLElement;
          fileInputEl.click();
        }
      }, [
        this.displayProgressBar(),

        m('div.upload-content',
          vnode.attrs.active ?
            this.displayUploadInstructions() :
            vnode.children
        )
      ]),

      m("input", {
        type: "file",
        accept: 'image/png, image/jpeg',
        id: "imageIn", 
        name: "imgFile",
        multiple: true, 
        hidden:"hidden", 
        onchange: async e => {
          await this.uploadImages(Array.from(e.target.files));
        }
      })
    ];
  }
}