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
        this.tags = projectInfo.tags;
        this.images = projectInfo.images;

        if (this.images.length > 0)
          this.fetchImages();

        if (this.tags.length > 0)
          this.colors = colorGen(this.tags.length);

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
        this.images = this.images.filter(img => !imageIds.includes(img._id));
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
          colors: this.colors,
    
          res: this.fetchImages.bind(this),
          addTag: this.addTag.bind(this),
          updatePopupStatus: this.popupManager.updatePopupStatus.bind(this.popupManager),
          unshiftSelection: this.unshiftSelection.bind(this),
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

              removeImages: this.removeImages.bind(this),
            }) : null 
        ),

      ]),

    ]);
  }
}