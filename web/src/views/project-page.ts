import m from 'mithril';
import colorGen from 'iwanthue';

import '../css/project-page.scss';
import '../css/image-list.scss';

import { fetchRequest, initArr } from '../utils/utils';
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
  private isAdmin: boolean;

  private projectId: string;

  private popupManager: PopupManager;
  
  private info: object;

  private tags: TagAttrs[];
  private colors: string[];
  private images: ImageAttrs[];

  private selectedImages: string[];
  private highlightedTags: Record<string, number>;

  constructor(vnode : m.CVnode<{projectId: string}>) {
    this.uid = Cookies.get('uid');
    this.isAdmin = false;

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
    this.highlightedTags = {};

    this.popupManager = new PopupManager();
    this.popupManager.setReloadCallback(this.fetchProject.bind(this));
    this.popupManager.linkPopup('editTag', EditTagPopup);

    this.fetchProject();
  }

  fetchProject() : void {
    if (!this.uid) return;
    console.log(this.uid);

    fetchRequest<ProjectAttrs>('/getProject', {
      method: 'GET',
      query: {
        projectId: this.projectId,
        uid: this.uid,
      },
    })
      .then(projectInfo => {
        console.log('projectInfo', projectInfo);

        this.isAdmin = projectInfo.patronId === this.uid;

        this.info = projectInfo;
        this.images = projectInfo.images.sort((a, b) => 
          Number(a.validated) - Number(b.validated)
        );

        if (this.images.length > 0)
          this.fetchImages();
        
        this.tags = projectInfo.tags;
        
        this.refreshTagColors();
        this.refreshHighlightedCategories();

        m.redraw();
      })
      .catch(console.error)
  }

  addTag() : void {
    if (!this.isAdmin) return;

    fetchRequest<TagAttrs>('/createTag', {
      method: 'POST',
      body: {
        projectId: this.projectId,
        uid: this.uid,
      }
    })
      .then(tag => {
        this.tags.push(tag);
        this.refreshTagColors();
        this.refreshHighlightedCategories();
        m.redraw();
      })
      .catch(console.log)
  }

  updateTag(tagId: string, tagUpdate: {
    name: string,
    description: string,
    goalQty: number,
  }): void {
    if (!this.isAdmin) return;

    console.log('updatetag', tagId, tagUpdate);
    fetchRequest('/updateTag', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        tagId,
        update: tagUpdate,
      }
    })
      .then(res => {
        Object.assign(
          this.tagById[tagId],
          tagUpdate,
        );
        m.redraw();
      })
  }

  removeTag(tagId: string): void {
    if (!this.isAdmin) return;

    fetchRequest('/removeTag', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        tagId,
      }
    })
      .then(res => {
        let i : number;

        // remove tag from tag list
        for (i = 0; i < this.tags.length; i++)
          if (this.tags[i]._id === tagId)
            this.tags.splice(i, 1);
        
        // remove tag from images that include it
        for (i = 0; i < this.images.length; i++) {
          let tagIndex = this.images[i].tags.indexOf(tagId);
          if (tagIndex !== -1)
            this.images[i].tags.splice(tagIndex, 1);
        }

        m.redraw();
      })
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

  refreshTagColors() : void {
    if (this.tags.length === 0) return;

    this.colors = colorGen(this.tags.length);
    this.tags = this.tags
      .map((tag: TagAttrs, i: number) => 
        Object.assign(tag, {
          color: this.colors[i],
        })
      );
  }

  refreshHighlightedCategories() : void {
    this.highlightedTags = {};
    let imgInfo = this.imageById;

    for (let i = 0; i < this.tags.length; i++) {
      let tagId = this.tags[i]._id;

      this.highlightedTags[tagId] = 0;

      if (this.selectedImages.length === 0)
        continue;

      for (let j = 0; j < this.selectedImages.length; j++) {
        let img = imgInfo[this.selectedImages[j]];
        if (img.tags.includes(tagId))
          this.highlightedTags[tagId]++;
      }

      this.highlightedTags[tagId] /= this.selectedImages.length;
    }
  }

  overrideHighlightedCategories(highlightedIds: string[]) : void {
    if (highlightedIds.length > 0) {
      this.highlightedTags = {};

      for (let tag of this.tags) {
        this.highlightedTags[tag._id] = 0;

        if (highlightedIds.includes(tag._id))
          this.highlightedTags[tag._id] = 1;
      }
    }
    else
      this.refreshHighlightedCategories();
  }

  updateImageSelection(selectedIds: string[]) : void {
    this.selectedImages = selectedIds;
    this.refreshHighlightedCategories();
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
        m.redraw();
      })
  }

  setValidationStatus(imageIds: string[], validate: boolean) : void {
    fetchRequest('/updateImages', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        imageIds,
        validate,
      }
    })
      .then(() => {
        for (let id of imageIds)
          this.imageById[id].validated = validate;
        
        m.redraw();
      })
  }

  applyTagToImages(tag: TagAttrs) {
    let isAdding = this.highlightedTags[tag._id] < 1;
    
    let body = {
      uid: this.uid,
      projectId: this.projectId,
      imageIds: this.selectedImages,
    };

    if (isAdding) body['addTags'] = [tag._id];
    else body['removeTags'] = [tag._id];

    fetchRequest<{modifiedCount: number}>('/updateImages', {
      method: 'POST',
      body,
    })
      .then(res => {
        let imgInfo = this.imageById;
        
        for (let imgId of this.selectedImages) {
          let img = imgInfo[imgId];

          if (!isAdding)
            img.tags.splice(img.tags.indexOf(tag._id), 1);
          else if (!img.tags.includes(tag._id))
            img.tags.push(tag._id);
        }
        
        this.refreshHighlightedCategories();
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
          isAdmin: this.isAdmin,
    
          categories: this.tags,
          activeCategories: this.highlightedTags,
          
          addTag: this.addTag.bind(this),
          updateTag: this.updateTag.bind(this),
          removeTag: this.removeTag.bind(this),

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
              isAdmin: this.isAdmin,
              uid: this.uid,

              images: this.images,
              tagById: this.tagById,

              removeImages: this.removeImages.bind(this),
              setValidationStatus: this.setValidationStatus.bind(this),
              changeImageSelection: this.updateImageSelection.bind(this),
              changeTagHighlight: this.overrideHighlightedCategories.bind(this),
            }) : null 
        ),

      ]),

    ]);
  }
}