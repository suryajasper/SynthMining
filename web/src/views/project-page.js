import m from 'mithril';
import '../css/project-page.scss';
import { fetchRequest, initArr } from '../utils/utils';
import Cookies from '../utils/Cookies';
import ImageUpload from './fileupload';
import ImageList from './img-view';
import colorGen from 'iwanthue';

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

class CategorySelections {
  constructor(vnode) {
    this.selected = [];
  }

  view(vnode) {
    if (this.selected.length != vnode.attrs.categories.length)
      this.selected = initArr(vnode.attrs.categories.length, true);
    
    return m('div.category-container',
      m('div.category-content',
        (vnode.attrs.categories || [])
          .map((cat, i) => 
            m('div.category-item', {
              class: this.selected[i] ? 'selected' : '',
              onclick: e => {
                if (this.selected[i-1] || !(this.selected[i])) {
                  this.selected = initArr(this.selected.length, false);
                  this.selected[i] = true;

                  vnode.attrs.res(cat);
                } else {
                  this.selected = initArr(this.selected.length, true);
                }
              }
            }, [
              m('span.category-item-color', {
                style: {
                  backgroundColor: vnode.attrs.colors[i]
                }
              }),
              m('span.category-item-title', cat.name),
            ])
          )
          .concat([
            m('div.category-item.new-category-button.selected', {
              onclick: e => {
                vnode.attrs.addTag();
              }
            }, [
              m('span.category-item-new-icon', '+'),
              m('span.category-item-title', 'Add Tag'),
            ])
          ])
      ),
    );
  }
}

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
      .then(tagInfo => {
        this.fetchProjectInfo();
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

  view(vnode) {
    return m('div.project-page', [

      m('div.left-container', [
        m(CategorySelections, {
          categories: this.tags,
          projectId: this.projectId,
          colors: this.colors,

          res: this.fetchImages.bind(this),
          addTag: this.addTag.bind(this),
        }),
      ]),

      this.images.length === 0 ? 
        m(ImageUpload, {
          active: true,
          uid: this.uid, 
          projectId: this.projectId,
          imgSrcs: this.images,
          status: e => {
            if (!e.err)
              this.fetchImages();
          } 
        }) : 
        m('div.right-container', [
          m(ImageList, {
            images: this.images
          })
        ])

    ]);
  }
}