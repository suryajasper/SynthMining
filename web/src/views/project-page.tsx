import React from 'react';
import m from 'mithril';
import colorGen from 'iwanthue';

import '../css/project-page.scss';
import '../css/image-list.scss';

import { fetchRequest, initArr } from '../utils/utils';
import Cookies from '../utils/Cookies';

import { EditTagPopup } from './popups/edit-tag-popup';
import { PopupOverlay } from './popups/popup';
import PopupManager from './popups/popup-manager';

import CategorySelections from './category-list';
import ImageUpload from './fileupload';
import ImageList from './img-view';

import { ImageAttrs, ProjectAttrs, TagAttrs } from './project-loader';

interface ProjectPageAttrs {
  projectId: string;
}

interface ProjectPageState {
  uid: string | undefined;
  isAdmin: boolean;

  projectId: string;
  
  info: object;
  
  tags: TagAttrs[];
  colors: string[];
  images: ImageAttrs[];
  
  selectedImages: string[];
  highlightedTags: Record<string, number>;
}

export default class ProjectPage 
  extends React.Component<ProjectPageAttrs, ProjectPageState> {
  
  private popupManager: PopupManager;

  constructor(props: ProjectPageAttrs) {
    super(props);
    
    // if (!this.uid) m.route.set('/login');
    this.popupManager = new PopupManager();
    this.popupManager.setReloadCallback(this.fetchProject.bind(this));
    this.popupManager.linkPopup('editTag', EditTagPopup);

    this.state = {
      uid: Cookies.get('uid'),
      isAdmin: false,

      projectId: this.props.projectId,
      
      info: {
        name: '',
        description: '',

        patronId: '',
        tags: [''],
      },

      tags : [],
      colors : [],
      images : [],

      selectedImages: [],
      highlightedTags: {},
    };

    this.fetchProject();
  }

  fetchProject() : void {
    if (!this.state.uid) return;
    console.log(this.state.uid);

    fetchRequest<ProjectAttrs>('/getProject', {
      method: 'GET',
      query: {
        projectId: this.props.projectId,
        uid: this.state.uid,
      },
    })
      .then(projectInfo => {
        console.log('projectInfo', projectInfo);

        this.setState({
          isAdmin: projectInfo.patronId === this.state.uid,
          info: projectInfo,
          images: projectInfo.images.sort((a, b) => 
            Number(a.validated) - Number(b.validated)
          ),
          tags: projectInfo.tags,
        }, () => {
          console.log('set state', this.state);
  
          if (this.state.images.length > 0)
            this.fetchImages();
          
          this.refreshTagColors();
          this.refreshHighlightedCategories();
        });
      })
      .catch(console.error)
  }

  addTag() : void {
    if (!this.state.isAdmin) return;

    fetchRequest<TagAttrs>('/createTag', {
      method: 'POST',
      body: {
        projectId: this.props.projectId,
        uid: this.state.uid,
      }
    })
      .then(async tag => {
        await this.setState(({tags}) => {
          tags.push(tag);
          return { tags };
        });

        this.refreshTagColors();
        this.refreshHighlightedCategories();
      })
      .catch(console.log)
  }

  updateTag(tagId: string, tagUpdate: {
    name: string,
    description: string,
    goalQty: number,
  }): void {
    if (!this.state.isAdmin) return;

    console.log('updatetag', tagId, tagUpdate);
    fetchRequest('/updateTag', {
      method: 'POST',
      body: {
        uid: this.state.uid,
        projectId: this.props.projectId,
        tagId,
        update: tagUpdate,
      }
    })
      .then(res => {
        this.setState(({tags}) => {
          for (let i = 0; i < tags.length; i++)
            if (tags[i]._id === tagId)
              Object.assign(tags[i], tagUpdate);
          
          return { tags }
        });
      })
  }

  removeTag(tagId: string): void {
    if (!this.state.isAdmin) return;

    fetchRequest('/removeTag', {
      method: 'POST',
      body: {
        uid: this.state.uid,
        projectId: this.props.projectId,
        tagId,
      }
    })
      .then(res => {
        this.setState(({tags, images}) => {
          let i : number;
          
          // remove tag from tag list
          for (i = 0; i < tags.length; i++)
            if (tags[i]._id === tagId)
              tags.splice(i, 1);
              
          // remove tag from images that include it
          for (i = 0; i < images.length; i++) {
            let tagIndex = images[i].tags.indexOf(tagId);

            if (tagIndex !== -1)
              images[i].tags.splice(tagIndex, 1);
          }
        });
      })
  }

  get tagById() : Record<string, TagAttrs> {
    let byId = {};

    for (let tag of this.state.tags)
      byId[tag._id] = tag;

    return byId;
  }

  get imageById() : Record<string, ImageAttrs> {
    let byId = {};

    for (let img of this.state.images)
      byId[img._id] = img;

    return byId;
  }

  refreshTagColors() : void {
    if (this.state.tags.length === 0) return;

    console.log('refreshing colors bitch');

    this.setState(prevState => {
      const newColors : string[] = colorGen(prevState.tags.length);
      return {
        colors: newColors,
        tags: prevState.tags 
          .map((tag: TagAttrs, i: number) => 
            Object.assign(tag, {
              color: newColors[i],
            })
          )
      };
    });
  }

  refreshHighlightedCategories() : void {
    let imgInfo = this.imageById;

    this.setState(state => {
      let highlightedTags = {};
  
      for (let i = 0; i < state.tags.length; i++) {
        let tagId = state.tags[i]._id;
  
        highlightedTags[tagId] = 0;
  
        if (state.selectedImages.length === 0)
          continue;
  
        for (let j = 0; j < state.selectedImages.length; j++) {
          let img = imgInfo[state.selectedImages[j]];
          if (img.tags.includes(tagId))
            highlightedTags[tagId]++;
        }
  
        highlightedTags[tagId] /= state.selectedImages.length;
      }

      return { highlightedTags };
    });
  }

  overrideHighlightedCategories(highlightedIds: string[]) : void {
    if (highlightedIds.length > 0)
      this.setState(state => {
        let highlightedTags = {};
  
        for (let tag of state.tags) {
          highlightedTags[tag._id] = 0;
  
          if (highlightedIds.includes(tag._id))
            highlightedTags[tag._id] = 1;
        }

        return { highlightedTags };
      });
    
    else
      this.refreshHighlightedCategories();
  }

  updateImageSelection(selectedIds: string[]) : void {
    this.setState(
      { selectedImages: selectedIds }, 
      this.refreshHighlightedCategories.bind(this)
    );
  }

  fetchImages() : void {
    let params = {
      img_count: this.state.images.length,
    };

    this.state.images.forEach((img, i) => {
      params[`id_${i}`] = img._id;
    });

    m.request<{images: string[]}>('http://localhost:2003/getImages', {
      method: 'GET',
      params,
    }).then(res => {
      this.setState(({images}) => ({
        images: images.map((imgData, i) => 
          Object.assign(imgData, {src: res.images[i]})
        ),
      }));
    }).catch(console.error)
  }

  removeImages(imageIds: string[]) : void {
    fetchRequest('/removeImages', {
      method: 'POST',
      body: {
        uid: this.state.uid,
        projectId: this.props.projectId,
        imageIds,
      },
    })
      .then(() => {
        this.setState(({images}) => ({
          images: images.filter(img => !imageIds.includes(img._id)),
        }));
      })
  }

  setValidationStatus(imageIds: string[], validate: boolean) : void {
    fetchRequest('/updateImages', {
      method: 'POST',
      body: {
        uid: this.state.uid,
        projectId: this.props.projectId,
        imageIds,
        validate,
      }
    })
      .then(() => {
        this.setState(({images}) => {
          for (let i = 0; i < images.length; i++)
            if (imageIds.includes(images[i]._id))
              images[i].validated = validate;
          
          return { images };
        });
      })
  }

  applyTagToImages(tag: TagAttrs) {
    let isAdding = this.state.highlightedTags[tag._id] < 1;
    
    let body = {
      uid: this.state.uid,
      projectId: this.props.projectId,
      imageIds: this.state.selectedImages,
    };

    if (isAdding) body['addTags'] = [tag._id];
    else body['removeTags'] = [tag._id];

    fetchRequest<{modifiedCount: number}>('/updateImages', {
      method: 'POST',
      body,
    })
      .then(async res => {
        await this.setState(prevState => {
          for (let img of prevState.images) {
            if (!prevState.selectedImages.includes(img._id))
              continue;
            
            if (!isAdding)
              img.tags.splice(img.tags.indexOf(tag._id), 1);
            else if (!img.tags.includes(tag._id))
              img.tags.push(tag._id);
          }

          return { images: prevState.images };
        });
        
        this.refreshHighlightedCategories();
      })
  }

  unshiftSelection(i: number) {
    this.setState(({tags}) => {
      let selectedTag = tags.splice(i, 1);
      tags.unshift(...selectedTag);
      return { tags };
    })
  }

  render() {
    return (
      <div className='project-page'>
        <PopupOverlay>
          <>{
            Object.values(this.popupManager.popups)
              .map((popup, i) => 
                React.createElement(
                  popup.view, 
                  Object.assign({key: `popup-${i}`}, popup.attrs)
                )
              )
          }</>
        </PopupOverlay>

        <div className='left-container'>
          <CategorySelections
            uid={this.state.uid}
            projectId={this.props.projectId}
            isAdmin={this.state.isAdmin}
      
            categories={this.state.tags}
            activeCategories={this.state.highlightedTags}
            
            addTag={this.addTag.bind(this)}
            updateTag={this.updateTag.bind(this)}
            removeTag={this.removeTag.bind(this)}
            
            unshiftSelection={this.unshiftSelection.bind(this)}

            applyToImageMode={this.state.selectedImages.length > 0}
            applyToImage={this.applyTagToImages.bind(this)}
          />
        </div>

        <div className='right-container'>
          <ImageUpload
            active={this.state.images.length === 0}
            uid={this.state.uid} 
            projectId={this.props.projectId}
            status={e => {
              if (!e.err)
                this.fetchProject();
            }}
          >
            <>{
              this.state.images.length > 0 && 
                <ImageList
                  isAdmin={this.state.isAdmin}
                  uid={this.state.uid}

                  images={this.state.images}
                  tagById={this.tagById}

                  removeImages={this.removeImages.bind(this)}
                  setValidationStatus={this.setValidationStatus.bind(this)}
                  changeImageSelection={this.updateImageSelection.bind(this)}
                  changeTagHighlight={this.overrideHighlightedCategories.bind(this)}
                />
            }</>
          </ImageUpload>
        </div>
      </div>
    );
  }
}