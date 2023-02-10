import m from 'mithril';
import colorGen from 'iwanthue';

import '../css/project-page.scss';
import '../css/image-list.scss';

import { fetchRequest } from '../utils/utils';
import Cookies from '../utils/Cookies';

import { EditTagPopup, CategorySelections} from './category-list';
import { PopupOverlay, updatePopupOverlayStatus } from './popup';
import ImageUpload from './fileupload';
import ImageList from './img-view';
/*
function ProjectInfo(vnode) {
  return {
    view(vnode) {
      return m('div.info-container', [
        m('span.project-title', vnode.attrs.name),

        m('div.project-description-container', [

          m('div.project-tags-list', vnode.attrs.tags.map(
            tag => m('div.project-tag-item', tag)
          )),

            m('div.project-description', vnode.attrs.description),

        ]),
      ]);
    },
  };
}
*/
export default class ProjectPage {
  constructor(vnode) {
    this.uid = Cookies.get('uid');
    this.projectId = vnode.attrs.projectId;

    if (!this.uid) m.route.set('/login');
    
    this.editing = false;
    this.info = {
      name: '',
      description: '',

      patronId: '',
      tags: [''],
    }
    this.tags = [];
    this.colors = [];
    this.images = [];

    this.popups = {
      'editTag': EditTagPopup,
    };

    Object.keys(this.popups).forEach(popupName => {
      const popupView = this.popups[popupName];
      this.popups[popupName] = {
        view: popupView,
        attrs: {
          active: true,
          data: {},

          disabledCallback: this.inactivateAllPopups.bind(this),
          reloadCallback: this.fetchProject.bind(this),
        },
      }
    });

    this.fetchProject();
  }

  fetchProject() {
    fetchRequest('/getProject', {
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

  addTag() {
    fetchRequest('/createTag', {
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

  fetchImages(cat) {
    let params = {
      img_count: this.images.length,
    };

    this.images.forEach((img, i) => {
      params[`id_${i}`] = img._id;
    })

    m.request('http://localhost:2003/getImages', {
      method: 'GET',
      params,
    }).then(res => {
      if (!res.err && res.images) {
        this.images = this.images.map((imgData, i) => 
          Object.assign(imgData, {src: res.images[i]})
        );
        m.redraw();
      }
    }).catch(console.error)
  }

  isPopupActive() {
    return Object.values(this.popups)
      .map(popup => popup.active)
      .reduce((a, b) => a + b);
  }

  inactivateAllPopups() {
    updatePopupOverlayStatus({ active: false });

    Object.values(this.popups).forEach(popup => {
      popup.active = false;
    })
  }

  updatePopupStatus({ name, active, data }) {
    if (this.isPopupActive() || !this.popups[name]) 
      return;
    
    updatePopupOverlayStatus({ active });
    this.popups[name].attrs.active = active;
    this.popups[name].attrs.data = data;
    console.log(this.popups);
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
          categories: this.tags,
          uid: this.uid,
          projectId: this.projectId,
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
          imgSrcs: this.images,
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