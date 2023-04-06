import React from 'react';
import { ImageAttrs, TagAttrs } from './project-loader';
import { Icon, icons } from './icons';
import { initArr } from '../utils/utils';

const base64ImgHeader = (imgData: string) => `data:image/jpg;base64,${imgData}`;

interface ISharedImageAttrs {
  tagById: Record<string, TagAttrs>;
  changeTagHighlight: (highlightedTagIds: string[]) => void; 
}

interface ImageDisplayAttrs extends ImageAttrs, ISharedImageAttrs {
  selected: boolean;
  obscured: boolean;
  select: (shiftKey?: boolean) => void;
}

class ImageView extends React.Component<ImageDisplayAttrs> {
  trimText(str : string) : string {
    let maxLen : number = 1000;

    if (str.length > maxLen)
      return str.substring(0, 10) + '...';

    return str;
  }

  render() {
    return (
      <div className= {[ 'img-view-container',
          this.props.validated ? 'img-validated' : '',
          this.props.selected ? 'selected' : '',
          this.props.obscured ? 'not-selected' : '',
        ].join(' ')}

        onClick={(e: React.MouseEvent) => {
          e.preventDefault();
          e.stopPropagation();

          this.props.select(e.shiftKey);
        }}

        onMouseEnter={ (e: React.MouseEvent) => {
          e.preventDefault();
          e.stopPropagation();

          this.props.changeTagHighlight(this.props.tags);
        } }

        onMouseLeave={ (e: React.MouseEvent) => {
          e.preventDefault();
          e.stopPropagation();

          this.props.changeTagHighlight([]);
        } }
      >
        <div className='img-view-content' title={this.props.name}>
          <img className='image-display' src={base64ImgHeader(this.props.src)} />
          <span className='image-title'>{this.trimText(this.props.name)}</span>
          <div className='image-tag-list'>{
            this.props.tags.map((tag, i) => 
              <span 
                className='image-tag-color'
                key={`img-tag${i}`}
                style={{ backgroundColor: this.props.tagById[tag].color }}
              />
            )
          }</div>
        </div>
      </div>
    )
  }
}

function IconButton(props: {
  title: string,
  icon: JSX.Element,
  onclick?: (e: React.MouseEvent) => void,
}) {
  return (
    <button className='button'
      onClick={(e: React.MouseEvent) => {
        e.stopPropagation();
        props.onclick(e);
      }}
    >
      {props.icon}
      <span>{props.title}</span>
    </button>
  );
}

interface ImageListAttrs extends ISharedImageAttrs {
  uid: string;

  images: Array<ImageAttrs>;
  isAdmin: boolean;
  
  removeImages: (imageIds: string[]) => Promise<any>;
  setValidationStatus: (imageIds: string[], validate: boolean) => void;
  changeImageSelection: (selectedIds: string[]) => void;
}

interface ImageListState {
  selected: boolean[];
  last: number;
}

export default class ImageList 
  extends React.Component<ImageListAttrs, ImageListState> {
  
  private iEnum: number;

  constructor(props: ImageListAttrs) {
    super(props);

    this.iEnum = 0;

    this.state = {
      selected: initArr<boolean>(this.props.images.length, false),
      last: 0,
    };
  }

  setSelection(i: number | number[], val: boolean) {
    return new Promise(resolve => {
      this.setState(state => {
        if (typeof i === 'number')
          state.selected[i] = val;
        else
          for (let j = i[0]; j < i[1]; j++)
            state.selected[j] = val;
        
        return state;
      }, () => { resolve(null); })
    });
  }

  componentDidUpdate() {
    if (this.props.images.length != this.state.selected.length)
      this.setState({
        selected: initArr<boolean>(this.props.images.length, false)
      });
  }

  clearSelection() : void {
    this.setSelection([0, this.state.selected.length], false)
      .then(() => {
        this.props.changeImageSelection(this.selectedIds);
      });
  }

  selectAll() : void {
    this.setSelection([0, this.state.selected.length], true)
      .then(() => {
        this.props.changeImageSelection(this.selectedIds);
      });
  }

  canSelect(i) : boolean {
    return this.props.isAdmin || (
      !this.props.images[i].validated && 
      this.props.images[i].authorId === this.props.uid
    );
  }

  async select(i: number, shiftKey?: boolean) : Promise<void> {
    let newVal : boolean = !this.state.selected[i]

    if (this.canSelect(i))
      await this.setSelection(i, newVal);
    
    if (shiftKey)
      for (let j = this.state.last; j < i; j++)
        await this.setSelection([this.state.last, i], this.canSelect(j) && newVal);

    this.setState({ last: i });

    this.props.changeImageSelection(this.selectedIds);
  }

  renderImageView(img, i) {
    return React.createElement(ImageView, Object.assign(img, {
      select: (shiftKey: boolean) => {
        this.select(i, shiftKey);
      },
      selected: this.state.selected[i],
      obscured: this.selectedCount > 0 && !this.state.selected[i],

      tagById: this.props.tagById,
      changeTagHighlight: this.props.changeTagHighlight,
    }));
  }

  renderImageListHeader() : JSX.Element {
    return (
      <div className={['img-list-header',
        this.selectedCount > 0 ? 'active' : '',
      ].join(' ')}>
        <IconButton 
          title='Clear Selection' 
          icon={icons.exit} 
          onclick={this.clearSelection.bind(this)} 
        />

        {this.isSelectionInvalidated ? 
          (<>
            <IconButton 
              title='Validate'
              icon={icons.subscription.bunny}
              onclick={() => {
                this.props.setValidationStatus(this.selectedIds, true);
                this.clearSelection();
              }}
            />
            <IconButton 
              title='Reject'
              icon={icons.trash}
              onclick={() => {
                this.props.removeImages(this.selectedIds);
                this.clearSelection();
              }}
            />
          </>) :
          (<>
            <IconButton title='Dispatch to GAN' icon={icons.build}/>
            <IconButton 
              title='Delete Images'
              icon={icons.trash}
              onclick={() => {
                this.props.removeImages(this.selectedIds);
                this.clearSelection();
              }}
            />
          </>)
        }
      </div>
    );
  }

  renderImageList(title: string, images: ImageAttrs[]) {
    return (
      <>
        <span className='img-list-title'>{title}</span>
        <div className='img-list'>{
          images.map(img => this.renderImageView(img, this.iEnum++))
        }</div>
      </>
    );
  }

  get selectedIds() : string[] {
    return this.props.images
      .map(img => img._id)
      .filter((_, i) => this.state.selected[i]);
  }

  get selectedCount() : number {
    let count = 0;
    for (let val of this.state.selected)
      if (val) count++;
    return count;
  }

  get isSelectionInvalidated() : boolean {
    for (let i = 0; i < this.state.selected.length; i++) {
      if (this.state.selected[i] && this.props.images[i].validated)
        return false;
    }
    return true;
  }
  
  render() {
    this.iEnum = 0;

    return (
      <>
        {this.renderImageListHeader()}

        <div className='img-list-container'>
          {this.renderImageList('Unvalidated', this.props.images.filter(img => !img.validated))}
          {this.renderImageList(  'Validated', this.props.images.filter(img =>  img.validated))}
        </div>
      </>
    )
  }
}