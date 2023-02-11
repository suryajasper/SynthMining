import m from 'mithril';
import colorGen from 'iwanthue';

import '../css/project-page.scss';
import '../css/image-list.scss';

import { fetchRequest } from '../utils/utils';
import Cookies from '../utils/Cookies';

import { EditTagPopup, CategorySelections, CategorySelectionsAttrs, EditTagPopupAttrs} from './category-list';
import { Popup, PopupAttrs, PopupOverlay, updatePopupOverlayStatus } from './popup';
import ImageUpload from './fileupload';
import ImageList from './img-view';
import { ImageAttrs, ProjectAttrs, ProjectBaseAttrs, TagAttrs } from './project-loader';

export default class ProjectPage implements m.ClassComponent<{projectId: string}> {
  private uid: string | undefined;
  private projectId: string;
  
  private info: object;

  private tags: TagAttrs[];
  private colors: string[];
  private images: ImageAttrs[];

  private popups: Record<string, {
    view: any,
    attrs: PopupAttrs,
  }>;

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

    this.initializePopups();    

    this.fetchProject();
  }

  initializePopups() : void {
    this.popups = {};

    const popupViews = {
      'editTag': EditTagPopup,
    }
    
    Object.keys(popupViews).forEach(popupName => {
      const popupView = popupViews[popupName];
      const attrs : EditTagPopupAttrs = {
        active: true,
        data: undefined,

        disabledCallback: this.inactivateAllPopups.bind(this),
        reloadCallback: this.fetchProject.bind(this),
      };

      this.popups[popupName] = {
        view: popupView,
        attrs,
      }
    });
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

  isPopupActive() : boolean {
    return Object.values(this.popups)
      .map(popup => popup.attrs.active)
      .reduce((a, b) => a || b);
  }

  inactivateAllPopups() : void {
    updatePopupOverlayStatus({ active: false });

    Object.values(this.popups).forEach(popup => {
      popup.attrs.active = false;
    })
  }

  updatePopupStatus({ name, active, data }) : void {
    if (this.isPopupActive() || !this.popups[name]) 
      return;
    
    updatePopupOverlayStatus({ active });
    this.popups[name].attrs.active = active;
    this.popups[name].attrs.data = data;
    m.redraw();
  }

  view(vnode) {

    return m('div.project-page', [

      m(PopupOverlay, {
        disabledCallback: this.inactivateAllPopups.bind(this),
      },
        Object.values(this.popups)
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
          updatePopupStatus: this.updatePopupStatus.bind(this),
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
              images: this.images
            }) : null 
        ),

      ]),

    ]);
  }
}