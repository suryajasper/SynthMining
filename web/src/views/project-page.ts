import m from 'mithril';
import colorGen from 'iwanthue';

import '../css/project-page.scss';
import '../css/image-list.scss';

import { fetchRequest } from '../utils/utils';
import Cookies from '../utils/Cookies';

import { EditTagPopup } from './popups/edit-tag-popup';
import { PopupOverlay } from './popups/popup';
import PopupManager from './popups/popup-manager';

import { CategorySelections } from './category-list';
import ImageUpload from './fileupload';
import ImageList from './img-view';

import { ImageAttrs, ProjectAttrs, TagAttrs } from './project-loader';

export default class ProjectPage implements m.ClassComponent<{projectId: string}> {
  private uid: string | undefined;
  private projectId: string;

  private popupManager: PopupManager;
  
  private info: object;

  private tags: TagAttrs[];
  private colors: string[];
  private images: ImageAttrs[];

  private selectedImages: string[];

  constructor(vnode : m.CVnode<{projectId: string}>) {
    this.uid = Cookies.get('uid');
    this.projectId = vnode.attrs.projectId;

    if (!this.uid) m.route.set('/login');
    
    this.info = {
      name: '',
      description: '',

      patronId: '',
      tags: [''],
    }
    this.tags = [];
    this.colors = [];
    this.images = [];

    this.selectedImages = [];

    this.popupManager = new PopupManager();
    this.popupManager.setReloadCallback(this.fetchProject);
    this.popupManager.linkPopup('editTag', EditTagPopup);

    this.fetchProject();
  }

  fetchProject() : void {
    if (!this.uid) return;

    fetchRequest<ProjectAttrs>('/getProject', {
      method: 'GET',
      query: {
        projectId: this.projectId,
        uid: this.uid,
      },
    })
      .then(projectInfo => {
        console.log('projectInfo', projectInfo);

        this.info = projectInfo;
        this.images = projectInfo.images;

        if (this.images.length > 0)
          this.fetchImages();

        if (projectInfo.tags.length > 0)
          this.colors = colorGen(projectInfo.tags.length);
        
        this.tags = projectInfo.tags
          .map((tag: TagAttrs, i: number) => 
            Object.assign(tag, {
              color: this.colors[i],
            })
          );

        m.redraw();
      })
      .catch(console.error)
  }

  addTag() : void {
    fetchRequest<TagAttrs>('/createTag', {
      method: 'POST',
      body: {
        projectId: this.projectId,
        uid: this.uid,
      }
    })
      .then(tag => { 
        this.tags.push(tag);
        this.colors = colorGen(this.tags.length);
        m.redraw();
      })
      .catch(console.log)
  }

  get tagById() : Record<string, TagAttrs> {
    let byId = {};

    for (let tag of this.tags)
      byId[tag._id] = tag;

    return byId;
  }

  get imageById() : Record<string, ImageAttrs> {
    let byId = {};

    for (let img of this.images)
      byId[img._id] = img;

    return byId;
  }

  fetchImages() : void {
    let params = {
      img_count: this.images.length,
    };

    this.images.forEach((img, i) => {
      params[`id_${i}`] = img._id;
    })

    m.request<{images: string[]}>('http://localhost:2003/getImages', {
      method: 'GET',
      params,
    }).then(res => {
      this.images = this.images.map((imgData, i) => 
        Object.assign(imgData, {src: res.images[i]})
      );
      m.redraw();
    }).catch(console.error)
  }

  removeImages(imageIds: string[]) : void {
    fetchRequest('/removeImages', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        imageIds,
      },
    })
      .then(() => {
        console.log('then');
        this.images = this.images.filter(img => !imageIds.includes(img._id));
        m.redraw();
      })
  }

  applyTagToImages(tag: TagAttrs) {
    console.log('sending request to apply tag', tag.name);
    fetchRequest<{updateCount: number}>('/updateImages', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        imageIds: this.selectedImages,
        addTags: [tag._id],
      }
    })
      .then(res => {
        console.log('applied tags', tag);
        for (let img of this.images) {
          if (this.selectedImages.includes(img._id)) {
            img.tags.push(tag._id);
          }
        }
        m.redraw();
      })
  }

  unshiftSelection(i: number) {
    let selectedTag = this.tags.splice(i, 1);
    this.tags.unshift(...selectedTag);
  }

  view(vnode) {

    return m('div.project-page', [

      m(PopupOverlay, {
        disabledCallback: this.popupManager.inactivateAllPopups.bind(this.popupManager),
      },
        Object.values(this.popupManager.popups)
          .map(popup => 
            m(popup.view, popup.attrs)
          )
      ),

      m('div.left-container', [
        m(CategorySelections, {
          uid: this.uid,
          projectId: this.projectId,
    
          categories: this.tags,
          
          addTag: this.addTag.bind(this),
          updatePopupStatus: this.popupManager.updatePopupStatus.bind(this.popupManager),
          unshiftSelection: this.unshiftSelection.bind(this),

          applyToImageMode: this.selectedImages.length > 0, 
          applyToImage: this.applyTagToImages.bind(this),
        }),
      ]),

      m('div.right-container', [

        m(ImageUpload, {
          active: this.images.length === 0,
          uid: this.uid, 
          projectId: this.projectId,
          status: e => {
            if (!e.err)
              this.fetchProject();
          } 
        },
          this.images.length > 0 ? 
            m(ImageList, {
              images: this.images,
              tagById: this.tagById,

              removeImages: this.removeImages.bind(this),
              changeImageSelection: selectedIds => {
                this.selectedImages = selectedIds;
              },
            }) : null 
        ),

      ]),

    ]);
  }
}